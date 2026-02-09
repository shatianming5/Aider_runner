# 精简报告（静态）

本文档：`docs/simplify_report.md`

该报告通过 AST 扫描 + 近似引用计数生成，用于帮助你在不破坏对外 API 与可审计性的前提下，减少私有符号面（private symbol surface）。

生成脚本与明细：

- 生成脚本：`scripts/simplify_report.py`（依赖 `scripts/annotate_symbols.py`）
- 明细 CSV：`docs/simplify_report.csv`

说明（Notes）：

- `refs≈`：对 `runner/` + `tests/` 做 regex 计数得到的近似引用次数（可能包含注释/字符串；可能漏掉反射/动态调用）。
- `action`：仅是建议；是否接受以 `pytest -q` 全量通过为准。

## 汇总（Summary）

- total symbols: 229
- by action:
  - SIMPLIFY: 69
  - KEEP: 160
- by simplifiable:
  - 是: 1
  - 部分: 71
  - 否: 157

## 按文件列出候选（non-KEEP）

下方只列出建议 `action != KEEP` 的符号；完整符号面请看 `docs/simplify_report.csv`。

### `runner/bootstrap.py`

- **SIMPLIFY** `run_bootstrap` [function] simp=部分 refs≈11 lines≈131
  - 原因：规模≈131 行；引用次数≈11（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/bootstrap.py:312; kind=function; refs≈11; lines≈131

### `runner/contract_repair.py`

- **SIMPLIFY** `repair_contract` [function] simp=部分 refs≈10 lines≈328
  - 原因：规模≈328 行；引用次数≈10（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/contract_repair.py:104; kind=function; refs≈10; lines≈328

### `runner/env.py`

- **SIMPLIFY** `EnvSession._evaluation` [method] simp=部分 refs≈1 lines≈198
  - 原因：规模≈198 行；引用次数≈1（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/env.py:654; kind=method; refs≈1; lines≈198
- **SIMPLIFY** `_inject_openai_base_compat` [function] simp=部分 refs≈2 lines≈22
  - 原因：规模≈22 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/env.py:105; kind=function; refs≈2; lines≈22
- **SIMPLIFY** `EnvSession._apply_llm_overrides` [method] simp=部分 refs≈2 lines≈24
  - 原因：规模≈24 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/env.py:443; kind=method; refs≈2; lines≈24
- **SIMPLIFY** `_runtime_openai_config` [function] simp=部分 refs≈2 lines≈26
  - 原因：规模≈26 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/env.py:124; kind=function; refs≈2; lines≈26
- **SIMPLIFY** `_hf_parquet_qa_rows` [function] simp=部分 refs≈2 lines≈34
  - 原因：规模≈34 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/env.py:152; kind=function; refs≈2; lines≈34
- **SIMPLIFY** `EnvSession._apply_runtime_env_inference_overrides` [method] simp=部分 refs≈3 lines≈21
  - 原因：规模≈21 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/env.py:499; kind=method; refs≈3; lines≈21
- **SIMPLIFY** `EnvSession._base_overrides` [method] simp=部分 refs≈3 lines≈31
  - 原因：规模≈31 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/env.py:467; kind=method; refs≈3; lines≈31
- **SIMPLIFY** `_resolve_llm` [function] simp=部分 refs≈3 lines≈38
  - 原因：规模≈38 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/env.py:51; kind=function; refs≈3; lines≈38
- **SIMPLIFY** `_validate_rollout_samples._norm_ws` [function] simp=部分 refs≈4 lines≈6
  - 原因：规模≈6 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/env.py:318; kind=function; refs≈4; lines≈6
- **SIMPLIFY** `EnvSession._audit_mode` [method] simp=部分 refs≈4 lines≈9
  - 原因：规模≈9 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/env.py:432; kind=method; refs≈4; lines≈9
- **SIMPLIFY** `_resolve_path` [function] simp=部分 refs≈5 lines≈6
  - 原因：规模≈6 行；引用次数≈5（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/env.py:38; kind=function; refs≈5; lines≈6
- **SIMPLIFY** `_resolve_run_root` [function] simp=部分 refs≈5 lines≈11
  - 原因：规模≈11 行；引用次数≈5（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/env.py:86; kind=function; refs≈5; lines≈11
- **SIMPLIFY** `_validate_rollout_samples` [function] simp=部分 refs≈10 lines≈165
  - 原因：规模≈165 行；引用次数≈10（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/env.py:252; kind=function; refs≈10; lines≈165

### `runner/env_local.py`

- **SIMPLIFY** `_classify_opencode_transport_error` [function] simp=部分 refs≈4 lines≈22
  - 原因：规模≈22 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/env_local.py:31; kind=function; refs≈4; lines≈22
- **SIMPLIFY** `_list_opencode_models` [function] simp=部分 refs≈4 lines≈32
  - 原因：规模≈32 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/env_local.py:147; kind=function; refs≈4; lines≈32
- **SIMPLIFY** `open_env` [function] simp=部分 refs≈9 lines≈287
  - 原因：规模≈287 行；引用次数≈9（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/env_local.py:273; kind=function; refs≈9; lines≈287

### `runner/eval_audit.py`

- **SIMPLIFY** `_looks_like_python_exec` [function] simp=部分 refs≈3 lines≈27
  - 原因：规模≈27 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/eval_audit.py:47; kind=function; refs≈3; lines≈27

### `runner/generic_rollout.py`

- **SIMPLIFY** `_maybe_rollout_hf_qa_parquet` [function] simp=部分 refs≈2 lines≈120
  - 原因：规模≈120 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/generic_rollout.py:225; kind=function; refs≈2; lines≈120
- **SIMPLIFY** `_resolve_openai_base` [function] simp=部分 refs≈3 lines≈31
  - 原因：规模≈31 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/generic_rollout.py:51; kind=function; refs≈3; lines≈31
- **SIMPLIFY** `_build_prompts` [function] simp=部分 refs≈3 lines≈36
  - 原因：规模≈36 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/generic_rollout.py:331; kind=function; refs≈3; lines≈36
- **SIMPLIFY** `_now_iso` [function] simp=部分 refs≈4 lines≈6
  - 原因：规模≈6 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/generic_rollout.py:43; kind=function; refs≈4; lines≈6
- **SIMPLIFY** `_norm_answer_str` [function] simp=部分 refs≈4 lines≈8
  - 原因：规模≈8 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/generic_rollout.py:188; kind=function; refs≈4; lines≈8
- **SIMPLIFY** `_extract_last_number` [function] simp=部分 refs≈4 lines≈9
  - 原因：规模≈9 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/generic_rollout.py:198; kind=function; refs≈4; lines≈9
- **SIMPLIFY** `_to_fraction` [function] simp=部分 refs≈4 lines≈10
  - 原因：规模≈10 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/generic_rollout.py:156; kind=function; refs≈4; lines≈10
- **SIMPLIFY** `_extract_final_line` [function] simp=部分 refs≈4 lines≈18
  - 原因：规模≈18 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/generic_rollout.py:168; kind=function; refs≈4; lines≈18
- **SIMPLIFY** `_norm_number_str` [function] simp=部分 refs≈5 lines≈6
  - 原因：规模≈6 行；引用次数≈5（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/generic_rollout.py:148; kind=function; refs≈5; lines≈6

### `runner/hints_exec.py`

- **SIMPLIFY** `_extract_score_from_text` [function] simp=部分 refs≈2 lines≈26
  - 原因：规模≈26 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/hints_exec.py:431; kind=function; refs≈2; lines≈26
- **SIMPLIFY** `_docker_available` [function] simp=部分 refs≈2 lines≈29
  - 原因：规模≈29 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/hints_exec.py:944; kind=function; refs≈2; lines≈29
- **SIMPLIFY** `normalize_hint_command._looks_like_fire_cli` [function] simp=部分 refs≈2 lines≈30
  - 原因：规模≈30 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/hints_exec.py:745; kind=function; refs≈2; lines≈30
- **SIMPLIFY** `_infer_repo_python_pin` [function] simp=部分 refs≈2 lines≈40
  - 原因：规模≈40 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/hints_exec.py:132; kind=function; refs≈2; lines≈40
- **SIMPLIFY** `run_hints._maybe_prepare_dataset_override` [function] simp=部分 refs≈2 lines≈142
  - 原因：规模≈142 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/hints_exec.py:1348; kind=function; refs≈2; lines≈142
- **SIMPLIFY** `_extract_invoked_command` [function] simp=部分 refs≈3 lines≈22
  - 原因：规模≈22 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/hints_exec.py:970; kind=function; refs≈3; lines≈22
- **SIMPLIFY** `normalize_hint_command._maybe_normalize_fire_flag_aliases` [function] simp=部分 refs≈3 lines≈22
  - 原因：规模≈22 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/hints_exec.py:775; kind=function; refs≈3; lines≈22
- **SIMPLIFY** `_hint_runtime_compatible` [function] simp=部分 refs≈3 lines≈23
  - 原因：规模≈23 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/hints_exec.py:536; kind=function; refs≈3; lines≈23
- **SIMPLIFY** `run_hints._parse_pytest_counts` [function] simp=部分 refs≈3 lines≈28
  - 原因：规模≈28 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/hints_exec.py:1485; kind=function; refs≈3; lines≈28
- **SIMPLIFY** `_extract_score_from_json_obj` [function] simp=部分 refs≈3 lines≈33
  - 原因：规模≈33 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/hints_exec.py:459; kind=function; refs≈3; lines≈33
- **SIMPLIFY** `normalize_hint_command._maybe_bound_openai_codegen_eval` [function] simp=部分 refs≈3 lines≈34
  - 原因：规模≈34 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/hints_exec.py:798; kind=function; refs≈3; lines≈34
- **SIMPLIFY** `normalize_hint_command._strip_pytest_xdist_flags` [function] simp=部分 refs≈3 lines≈39
  - 原因：规模≈39 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/hints_exec.py:839; kind=function; refs≈3; lines≈39
- **SIMPLIFY** `_is_remote_openai_hint` [function] simp=部分 refs≈4 lines≈6
  - 原因：规模≈6 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/hints_exec.py:401; kind=function; refs≈4; lines≈6
- **SIMPLIFY** `_canonical_base_url` [function] simp=部分 refs≈4 lines≈12
  - 原因：规模≈12 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/hints_exec.py:325; kind=function; refs≈4; lines≈12
- **SIMPLIFY** `_hint_backend` [function] simp=部分 refs≈4 lines≈13
  - 原因：规模≈13 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/hints_exec.py:386; kind=function; refs≈4; lines≈13
- **SIMPLIFY** `_normalize_score` [function] simp=部分 refs≈4 lines≈14
  - 原因：规模≈14 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/hints_exec.py:414; kind=function; refs≈4; lines≈14
- **SIMPLIFY** `run_hints._looks_like_openai_codegen_eval` [function] simp=部分 refs≈4 lines≈19
  - 原因：规模≈19 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/hints_exec.py:1321; kind=function; refs≈4; lines≈19
- **SIMPLIFY** `run_hints._exec` [function] simp=部分 refs≈4 lines≈27
  - 原因：规模≈27 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/hints_exec.py:1726; kind=function; refs≈4; lines≈27
- **SIMPLIFY** `run_hints._priority` [function] simp=部分 refs≈4 lines≈38
  - 原因：规模≈38 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/hints_exec.py:1270; kind=function; refs≈4; lines≈38
- **SIMPLIFY** `_extract_cli_flag_value_any` [function] simp=部分 refs≈5 lines≈10
  - 原因：规模≈10 行；引用次数≈5（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/hints_exec.py:374; kind=function; refs≈5; lines≈10
- **SIMPLIFY** `_as_major_minor` [function] simp=部分 refs≈5 lines≈19
  - 原因：规模≈19 行；引用次数≈5（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/hints_exec.py:110; kind=function; refs≈5; lines≈19
- **SIMPLIFY** `_matched_anchors` [function] simp=部分 refs≈5 lines≈20
  - 原因：规模≈20 行；引用次数≈5（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/hints_exec.py:1110; kind=function; refs≈5; lines≈20
- **SIMPLIFY** `_extract_cli_flag_value` [function] simp=部分 refs≈5 lines≈21
  - 原因：规模≈21 行；引用次数≈5（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/hints_exec.py:351; kind=function; refs≈5; lines≈21

### `runner/opencode_client.py`

- **SIMPLIFY** `OpenCodeClient._stop_local_server` [method] simp=部分 refs≈2 lines≈40
  - 原因：规模≈40 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/opencode_client.py:262; kind=method; refs≈2; lines≈40
- **SIMPLIFY** `OpenCodeClient._post_message_with_retry` [method] simp=部分 refs≈4 lines≈156
  - 原因：规模≈156 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/opencode_client.py:497; kind=method; refs≈4; lines≈156

### `runner/opencode_tooling.py`

- **SIMPLIFY** `_find_tag_gt` [function] simp=部分 refs≈2 lines≈32
  - 原因：规模≈32 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/opencode_tooling.py:350; kind=function; refs≈2; lines≈32
- **SIMPLIFY** `_extract_json_object` [function] simp=部分 refs≈2 lines≈35
  - 原因：规模≈35 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/opencode_tooling.py:313; kind=function; refs≈2; lines≈35
- **SIMPLIFY** `_sanitized_env` [function] simp=部分 refs≈3 lines≈24
  - 原因：规模≈24 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/opencode_tooling.py:663; kind=function; refs≈3; lines≈24
- **SIMPLIFY** `_is_env_like` [function] simp=部分 refs≈4 lines≈20
  - 原因：规模≈20 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/opencode_tooling.py:624; kind=function; refs≈4; lines≈20
- **SIMPLIFY** `_decode_attr_value` [function] simp=部分 refs≈4 lines≈26
  - 原因：规模≈26 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/opencode_tooling.py:88; kind=function; refs≈4; lines≈26
- **SIMPLIFY** `execute_tool_calls` [function] simp=部分 refs≈8 lines≈286
  - 原因：规模≈286 行；引用次数≈8（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/opencode_tooling.py:867; kind=function; refs≈8; lines≈286

### `runner/pipeline_verify.py`

- **SIMPLIFY** `run_pipeline_verification._teardown_allowed` [function] simp=部分 refs≈2 lines≈21
  - 原因：规模≈21 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/pipeline_verify.py:287; kind=function; refs≈2; lines≈21
- **SIMPLIFY** `run_pipeline_verification._env_get` [function] simp=部分 refs≈3 lines≈22
  - 原因：规模≈22 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/pipeline_verify.py:327; kind=function; refs≈3; lines≈22

### `runner/prompts.py`

- **SIMPLIFY** `make_scaffold_contract_prompt` [function] simp=部分 refs≈10 lines≈239
  - 原因：规模≈239 行；引用次数≈10（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/prompts.py:105; kind=function; refs≈10; lines≈239

### `runner/repo_resolver.py`

- **SIMPLIFY** `_repo_slug` [function] simp=部分 refs≈2 lines≈22
  - 原因：规模≈22 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/repo_resolver.py:96; kind=function; refs≈2; lines≈22
- **SIMPLIFY** `_parse_hf_dataset` [function] simp=部分 refs≈2 lines≈26
  - 原因：规模≈26 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/repo_resolver.py:237; kind=function; refs≈2; lines≈26
- **SIMPLIFY** `_extract_github_zip` [function] simp=部分 refs≈2 lines≈27
  - 原因：规模≈27 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/repo_resolver.py:208; kind=function; refs≈2; lines≈27
- **SIMPLIFY** `_hf_dataset_api_info` [function] simp=部分 refs≈2 lines≈27
  - 原因：规模≈27 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/repo_resolver.py:271; kind=function; refs≈2; lines≈27
- **SIMPLIFY** `_parse_github_owner_repo` [function] simp=部分 refs≈2 lines≈36
  - 原因：规模≈36 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/repo_resolver.py:120; kind=function; refs≈2; lines≈36
- **SIMPLIFY** `_download_hf_dataset_snapshot` [function] simp=部分 refs≈2 lines≈128
  - 原因：规模≈128 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/repo_resolver.py:305; kind=function; refs≈2; lines≈128
- **SIMPLIFY** `_default_clones_base` [function] simp=部分 refs≈3 lines≈25
  - 原因：规模≈25 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - 证据：runner/repo_resolver.py:26; kind=function; refs≈3; lines≈25
