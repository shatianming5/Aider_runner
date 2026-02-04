from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _lazy_import_ml():
    """中文说明：
    - 含义：延迟导入可选 ML 依赖（torch/transformers/peft）。
    - 内容：在用户实际运行训练命令时才导入；若缺失则给出可执行的安装提示。
    - 可简略：否（避免把重依赖强绑定到 runner 的基础安装/测试环境）。
    """
    try:
        import torch  # type: ignore
        from peft import LoraConfig, get_peft_model  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing ML dependencies. Install with:\n"
            "  pip install -r requirements-ml.txt\n"
            "(or install torch/transformers/peft manually for your platform)"
        ) from e
    return torch, AutoModelForCausalLM, AutoTokenizer, LoraConfig, get_peft_model


def _now_iso() -> str:
    """中文说明：
    - 含义：生成一个 ISO8601 时间戳字符串（本地时间）。
    - 内容：用于 metrics JSON 的 `ts` 字段，便于对齐 artifacts 时间线。
    - 可简略：是（纯工具函数；可直接用 time.strftime 拼接）。
    """
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def _pick_device(torch: Any, device: str) -> str:
    """中文说明：
    - 含义：根据用户输入选择训练/推理设备（cpu/cuda/mps/auto）。
    - 内容：auto 优先 cuda，其次 mps，最后 cpu；并对非法输入报错。
    - 可简略：可能（也可把逻辑直接写在 main；但抽出来更易测/复用）。
    """
    d = str(device or "auto").strip().lower() or "auto"
    if d == "auto":
        if bool(getattr(torch.cuda, "is_available", lambda: False)()):  # pragma: no cover
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and bool(getattr(mps, "is_available", lambda: False)()):  # pragma: no cover
            return "mps"
        return "cpu"
    if d in ("cpu", "cuda", "mps"):
        return d
    raise ValueError(f"invalid --device: {device} (expected auto/cpu/cuda/mps)")


def _infer_lora_target_modules(torch: Any, model: Any) -> list[str]:
    """中文说明：
    - 含义：从模型结构推断一个“尽量通用”的 LoRA target_modules 列表。
    - 内容：
      - 优先使用 Llama/Qwen 常见的投影层名（q/k/v/o/gate/up/down_proj）
      - 若不存在，则从 Linear 层的末级名字中选出现频率最高的若干个（避免空匹配）
    - 可简略：可能（这是便利性逻辑；用户也可显式传 --target-modules）。
    """
    preferred = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # Collect last-name stats for Linear layers.
    counts: dict[str, int] = {}
    for name, module in getattr(model, "named_modules", lambda: [])():
        if isinstance(module, getattr(torch.nn, "Linear")):
            last = str(name).split(".")[-1]
            counts[last] = counts.get(last, 0) + 1

    present_preferred = [m for m in preferred if counts.get(m)]
    if present_preferred:
        return present_preferred

    # Fall back to the most common Linear layer names.
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _v in ranked[:6]]


def _build_sft_texts() -> list[str]:
    """中文说明：
    - 含义：提供一个极小的、与任何 benchmark repo 无关的 SFT 文本样本集。
    - 内容：用于做少量 step 的 LoRA 微调，验证“训练 → 保存 → 推理”闭环可跑通。
    - 可简略：可能（可外部传入数据；但内置样本让脚本零依赖即可运行）。
    """
    pairs = [
        ("把 2 + 3 的结果只输出成一个数字。", "5"),
        ("把 10 - 7 的结果只输出成一个数字。", "3"),
        ("用一句话解释什么是单元测试。", "单元测试是在最小功能单元层面验证代码行为是否符合预期的自动化测试。"),
        ("给出 Python 里一个最小的函数定义示例。", "def f():\n    return 1"),
    ]
    out: list[str] = []
    for inst, resp in pairs:
        out.append(
            "\n".join(
                [
                    "### Instruction:",
                    inst,
                    "",
                    "### Response:",
                    resp,
                    "",
                ]
            )
        )
    return out


@dataclass(frozen=True)
class TrainResult:
    """中文说明：
    - 含义：训练脚本输出的结构化结果（也会写入 out_dir/train_metrics.json）。
    - 内容：包含是否成功、步数、耗时、最后 loss、模型路径等，便于 matrix/benchmark 端引用。
    - 可简略：是（也可直接用 dict；保留 dataclass 便于类型与序列化）。
    """

    ok: bool
    steps: int
    wall_time_s: float
    last_loss: float | None
    model_dir: str


def train_lora(
    *,
    base_model: str,
    out_dir: Path,
    steps: int,
    seq_len: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    seed: int,
    device: str,
    target_modules: list[str] | None,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> TrainResult:
    """中文说明：
    - 含义：对一个 ~0.5B base 模型做极小步数的 LoRA 微调，并导出可推理的 HF 模型目录。
    - 内容：下载 tokenizer+model → 注入 LoRA → 训练 N steps → merge_and_unload → save_pretrained(out_dir)。
    - 可简略：否（这是“训练 0.5B + 产物可用于 rollout/eval/benchmark”的核心实现）。
    """
    torch, AutoModelForCausalLM, AutoTokenizer, LoraConfig, get_peft_model = _lazy_import_ml()

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    d = _pick_device(torch, device)
    torch.manual_seed(int(seed or 0))

    # Prefer fp16 on accelerators; keep fp32 on CPU for compatibility.
    torch_dtype = torch.float16 if d in ("cuda", "mps") else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if getattr(tokenizer, "pad_token", None) is None:
        # Make tiny SFT workable for models without explicit pad token.
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.to(d)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        # Transformers commonly require disabling kv-cache for checkpointing.
        try:
            if getattr(getattr(model, "config", None), "use_cache", None) is not None:
                model.config.use_cache = False  # type: ignore[attr-defined]
        except Exception:
            pass

    modules = target_modules or _infer_lora_target_modules(torch, model)
    if not modules:
        raise RuntimeError("failed_to_infer_lora_target_modules (pass --target-modules)")

    lora_cfg = LoraConfig(
        r=int(lora_r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(modules),
    )
    peft_model = get_peft_model(model, lora_cfg)
    peft_model.train()

    # Tiny in-repo training texts (no dependency on any benchmark repo).
    texts = _build_sft_texts()
    if not texts:
        raise RuntimeError("no_training_texts")

    optim = torch.optim.AdamW(peft_model.parameters(), lr=float(lr))
    last_loss: float | None = None

    start = time.time()
    optim.zero_grad(set_to_none=True)
    for step_idx in range(int(steps)):
        text = texts[step_idx % len(texts)]
        batch = tokenizer(
            [text] * int(max(1, batch_size)),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(seq_len),
        )
        batch = {k: v.to(d) for k, v in batch.items()}
        out = peft_model(**batch, labels=batch["input_ids"])
        loss = out.loss
        last_loss = float(loss.detach().cpu().item()) if loss is not None else None
        (loss / float(max(1, grad_accum))).backward()
        if (step_idx + 1) % int(max(1, grad_accum)) == 0:
            optim.step()
            optim.zero_grad(set_to_none=True)

    wall = float(time.time() - start)

    # Export: merge LoRA weights into the base model and save a standalone HF directory.
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

    metrics = {
        "ts": _now_iso(),
        "train": {
            "ok": True,
            "steps": int(steps),
            "wall_time_s": wall,
            "loss": last_loss,
            "device": d,
            "base_model": base_model,
            "target_modules": list(modules),
            "lora": {"r": int(lora_r), "alpha": int(lora_alpha), "dropout": float(lora_dropout)},
        },
        "model_dir": str(out_dir),
    }
    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return TrainResult(ok=True, steps=int(steps), wall_time_s=wall, last_loss=last_loss, model_dir=str(out_dir))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """中文说明：
    - 含义：解析训练脚本的命令行参数。
    - 内容：把用户输入映射到 train_lora 的参数集合，并提供合理默认值（偏向“CPU 也能跑通”的小规模设置）。
    - 可简略：可能（也可直接在 main 里写 argparse；但抽出来更清晰）。
    """
    p = argparse.ArgumentParser(description="Train a small LoRA adapter and export a merged HF model directory.")
    p.add_argument("--base-model", required=True, help="Hugging Face model id or local path (e.g. Qwen/Qwen2.5-0.5B-Instruct)")
    p.add_argument("--out-dir", required=True, help="Output directory for merged model (HF save_pretrained format)")
    p.add_argument("--steps", type=int, default=int(os.environ.get("AIDER_FSM_TRAIN_STEPS") or 8))
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    p.add_argument(
        "--target-modules",
        default="",
        help="Comma-separated LoRA target module names (empty=auto infer). Example: q_proj,k_proj,v_proj,o_proj",
    )
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: list[str] | None = None) -> int:
    """中文说明：
    - 含义：训练脚本入口：LoRA 微调并导出 merged 模型目录。
    - 内容：解析 args → 调用 train_lora → 打印结果路径；失败时打印错误并返回非 0。
    - 可简略：否（CLI 入口；与用户直接交互）。
    """
    args = _parse_args(argv)
    targets = [s.strip() for s in str(args.target_modules or "").split(",") if s.strip()]
    try:
        res = train_lora(
            base_model=str(args.base_model),
            out_dir=Path(str(args.out_dir)),
            steps=int(args.steps),
            seq_len=int(args.seq_len),
            batch_size=int(args.batch_size),
            grad_accum=int(args.grad_accum),
            lr=float(args.lr),
            seed=int(args.seed),
            device=str(args.device),
            target_modules=targets or None,
            lora_r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
        )
        print(json.dumps({"ok": res.ok, "model_dir": res.model_dir, "steps": res.steps, "loss": res.last_loss}))
        return 0
    except Exception as e:
        print(f"ERROR: train_lora failed: {e}")
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
