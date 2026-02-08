from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _leading_ws(line: str) -> str:
    # 作用：内部符号：_leading_ws
    # 能否简略：是
    # 原因：测试代码（优先可读性）；规模≈5 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=tests/test_symbol_annotations_present.py:12；类型=function；引用≈3；规模≈5行
    i = 0
    while i < len(line) and line[i] in (" ", "\t"):
        i += 1
    return line[:i]


def _iter_py_files(base: Path) -> list[Path]:
    # 作用：内部符号：_iter_py_files
    # 能否简略：是
    # 原因：测试代码（优先可读性）；规模≈15 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=tests/test_symbol_annotations_present.py:19；类型=function；引用≈2；规模≈15行
    out: list[Path] = []
    if not base.exists():
        return out
    for p in base.rglob("*.py"):
        try:
            if not p.is_file():
                continue
        except Exception:
            continue
        if "__pycache__" in p.parts:
            continue
        out.append(p.resolve())
    out.sort()
    return out


@dataclass(frozen=True)
class _Target:
    # 作用：内部符号：_Target
    # 能否简略：部分
    # 原因：测试代码（优先可读性）；规模≈4 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=tests/test_symbol_annotations_present.py:37；类型=class；引用≈4；规模≈4行
    qualname: str
    indent: str
    start_idx: int
    end_idx: int


def _collect_targets(path: Path, text: str) -> list[_Target]:
    # 作用：内部符号：_collect_targets
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈70 行；引用次数≈2（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_symbol_annotations_present.py:43；类型=function；引用≈2；规模≈70行
    mod = ast.parse(text, filename=str(path))
    lines = text.splitlines(keepends=True)
    out: list[_Target] = []

    class _V(ast.NodeVisitor):
        # 作用：内部符号：_collect_targets._V
        # 能否简略：否
        # 原因：测试代码（优先可读性）；规模≈62 行；引用次数≈2（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
        # 证据：位置=tests/test_symbol_annotations_present.py:48；类型=class；引用≈2；规模≈62行
        def __init__(self) -> None:
            # 作用：内部符号：_collect_targets._V.__init__
            # 能否简略：是
            # 原因：测试代码（优先可读性）；规模≈3 行；引用次数≈1（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
            # 证据：位置=tests/test_symbol_annotations_present.py:49；类型=method；引用≈1；规模≈3行
            self.stack: list[str] = []
            self.class_stack: list[bool] = []

        def _push(self, name: str, *, is_class: bool) -> None:
            # 作用：内部符号：_collect_targets._V._push
            # 能否简略：是
            # 原因：测试代码（优先可读性）；规模≈3 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
            # 证据：位置=tests/test_symbol_annotations_present.py:53；类型=method；引用≈3；规模≈3行
            self.stack.append(name)
            self.class_stack.append(bool(is_class))

        def _pop(self) -> None:
            # 作用：内部符号：_collect_targets._V._pop
            # 能否简略：是
            # 原因：测试代码（优先可读性）；规模≈3 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
            # 证据：位置=tests/test_symbol_annotations_present.py:57；类型=method；引用≈3；规模≈3行
            self.stack.pop()
            self.class_stack.pop()

        def _qual(self, name: str) -> str:
            # 作用：内部符号：_collect_targets._V._qual
            # 能否简略：是
            # 原因：测试代码（优先可读性）；规模≈2 行；引用次数≈1（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
            # 证据：位置=tests/test_symbol_annotations_present.py:61；类型=method；引用≈1；规模≈2行
            return ".".join(self.stack + [name]) if self.stack else name

        def _is_method(self) -> bool:
            # 作用：内部符号：_collect_targets._V._is_method
            # 能否简略：是
            # 原因：测试代码（优先可读性）；规模≈2 行；引用次数≈0（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
            # 证据：位置=tests/test_symbol_annotations_present.py:64；类型=method；引用≈0；规模≈2行
            return bool(self.class_stack and self.class_stack[-1] is True)

        def _add(self, node: ast.AST, *, name: str) -> None:
            # 作用：内部符号：_collect_targets._V._add
            # 能否简略：部分
            # 原因：测试代码（优先可读性）；规模≈25 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
            # 证据：位置=tests/test_symbol_annotations_present.py:67；类型=method；引用≈3；规模≈25行
            if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
                return
            body = getattr(node, "body", None)
            if not isinstance(body, list) or not body:
                return
            first_stmt = body[0]

            doc_expr = None
            if (
                isinstance(first_stmt, ast.Expr)
                and isinstance(getattr(first_stmt, "value", None), ast.Constant)
                and isinstance(first_stmt.value.value, str)
            ):
                doc_expr = first_stmt

            if doc_expr is not None:
                end = int(getattr(doc_expr, "end_lineno", doc_expr.lineno))
                indent = _leading_ws(lines[int(doc_expr.lineno) - 1]) if int(doc_expr.lineno) - 1 < len(lines) else ""
                start_idx = max(0, end)
                if len(body) >= 2:
                    next_stmt = body[1]
                    end_idx = max(start_idx, int(getattr(next_stmt, "lineno")) - 1)
                else:
                    end_idx = min(len(lines), start_idx + 50)
            else:
                indent = (
                    _leading_ws(lines[int(getattr(first_stmt, "lineno")) - 1])
                    if int(getattr(first_stmt, "lineno")) - 1 < len(lines)
                    else ""
                )
                # Search between the header line(s) and the first statement.
                start_idx = max(0, int(getattr(node, "lineno")))
                end_idx = max(start_idx, int(getattr(first_stmt, "lineno")) - 1)

            out.append(_Target(qualname=self._qual(name), indent=indent, start_idx=int(start_idx), end_idx=int(end_idx)))

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            # 作用：内部符号：_collect_targets._V.visit_ClassDef
            # 能否简略：是
            # 原因：测试代码（优先可读性）；规模≈5 行；引用次数≈0（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
            # 证据：位置=tests/test_symbol_annotations_present.py:93；类型=method；引用≈0；规模≈5行
            self._add(node, name=str(node.name))
            self._push(str(node.name), is_class=True)
            self.generic_visit(node)
            self._pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            # 作用：内部符号：_collect_targets._V.visit_FunctionDef
            # 能否简略：是
            # 原因：测试代码（优先可读性）；规模≈5 行；引用次数≈0（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
            # 证据：位置=tests/test_symbol_annotations_present.py:99；类型=method；引用≈0；规模≈5行
            self._add(node, name=str(node.name))
            self._push(str(node.name), is_class=False)
            self.generic_visit(node)
            self._pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            # 作用：内部符号：_collect_targets._V.visit_AsyncFunctionDef
            # 能否简略：是
            # 原因：测试代码（优先可读性）；规模≈5 行；引用次数≈0（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
            # 证据：位置=tests/test_symbol_annotations_present.py:105；类型=method；引用≈0；规模≈5行
            self._add(node, name=str(node.name))
            self._push(str(node.name), is_class=False)
            self.generic_visit(node)
            self._pop()

    _V().visit(mod)
    return out


def _has_block(lines: list[str], *, start_idx: int, end_idx: int, indent: str) -> bool:
    # 作用：内部符号：_has_block
    # 能否简略：是
    # 原因：测试代码（优先可读性）；规模≈16 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=tests/test_symbol_annotations_present.py:115；类型=function；引用≈2；规模≈16行
    start_idx = max(0, int(start_idx))
    end_idx = min(len(lines), max(start_idx, int(end_idx)))
    for start in range(start_idx, min(len(lines), end_idx + 1)):
        if lines[start].startswith(indent + "# 作用："):
            needed = (
                indent + "# 作用：",
                indent + "# 能否简略：",
                indent + "# 原因：",
                indent + "# 证据：",
            )
            for off, prefix in enumerate(needed):
                if start + off >= len(lines):
                    return False
                if not lines[start + off].startswith(prefix):
                    return False
            return True
    return False


def test_all_symbols_have_annotations() -> None:
    # 作用：pytest 测试用例：验证行为契约
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈11 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_symbol_annotations_present.py:133；类型=function；引用≈1；规模≈11行
    offenders: list[str] = []
    for base in (ROOT / "runner", ROOT / "tests"):
        for p in _iter_py_files(base):
            text = p.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines(keepends=True)
            for t in _collect_targets(p, text):
                if not _has_block(lines, start_idx=t.start_idx, end_idx=t.end_idx, indent=t.indent):
                    offenders.append(f"{p.relative_to(ROOT)}::{t.qualname}")

    assert offenders == []
