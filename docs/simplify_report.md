# Simplification Report (static)
Generated via AST scan + approximate reference counting.

Notes:
- `refs≈` is a regex count across `runner/` + `tests/` (may include comments/strings; may miss reflection).
- `action` is a suggestion only; acceptance is `pytest -q`.

## Summary
- total symbols: 174
- by action:
  - SIMPLIFY: 32
  - KEEP: 142
- by simplifiable:
  - 是: 1
  - 部分: 34
  - 否: 139

## Per-file Candidates (non-KEEP)
Only symbols with suggested action != `KEEP` are listed below. Full surface is in CSV.

### `runner/bootstrap.py`

- **SIMPLIFY** `run_bootstrap` [function] simp=部分 refs≈11 lines≈131
  - reason: 规模≈131 行；引用次数≈11（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/bootstrap.py:312; kind=function; refs≈11; lines≈131

### `runner/contract_repair.py`

- **SIMPLIFY** `repair_contract` [function] simp=部分 refs≈10 lines≈328
  - reason: 规模≈328 行；引用次数≈10（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/contract_repair.py:102; kind=function; refs≈10; lines≈328

### `runner/env.py`

- **SIMPLIFY** `_hf_parquet_qa_rows` [function] simp=部分 refs≈2 lines≈34
  - reason: 规模≈34 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/env.py:39; kind=function; refs≈2; lines≈34
- **SIMPLIFY** `_validate_rollout_samples` [function] simp=部分 refs≈9 lines≈158
  - reason: 规模≈158 行；引用次数≈9（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/env.py:139; kind=function; refs≈9; lines≈158
- **SIMPLIFY** `EnvSession` [class] simp=部分 refs≈9 lines≈712
  - reason: 规模≈712 行；引用次数≈9（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/env.py:286; kind=class; refs≈9; lines≈712

### `runner/env_local.py`

- **SIMPLIFY** `open_env` [function] simp=部分 refs≈8 lines≈354
  - reason: 规模≈354 行；引用次数≈8（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/env_local.py:235; kind=function; refs≈8; lines≈354

### `runner/generic_rollout.py`

- **SIMPLIFY** `_maybe_rollout_hf_qa_parquet` [function] simp=部分 refs≈2 lines≈162
  - reason: 规模≈162 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/generic_rollout.py:114; kind=function; refs≈2; lines≈162

### `runner/hints_exec.py`

- **SIMPLIFY** `_extract_score_from_text` [function] simp=部分 refs≈2 lines≈26
  - reason: 规模≈26 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:424; kind=function; refs≈2; lines≈26
- **SIMPLIFY** `_docker_available` [function] simp=部分 refs≈2 lines≈29
  - reason: 规模≈29 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:891; kind=function; refs≈2; lines≈29
- **SIMPLIFY** `_extract_score_from_json_obj` [function] simp=部分 refs≈2 lines≈34
  - reason: 规模≈34 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:452; kind=function; refs≈2; lines≈34
- **SIMPLIFY** `_infer_repo_python_pin` [function] simp=部分 refs≈2 lines≈40
  - reason: 规模≈40 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:125; kind=function; refs≈2; lines≈40
- **SIMPLIFY** `_extract_invoked_command` [function] simp=部分 refs≈3 lines≈22
  - reason: 规模≈22 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:917; kind=function; refs≈3; lines≈22
- **SIMPLIFY** `_hint_runtime_compatible` [function] simp=部分 refs≈3 lines≈23
  - reason: 规模≈23 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:530; kind=function; refs≈3; lines≈23
- **SIMPLIFY** `_is_remote_openai_hint` [function] simp=部分 refs≈4 lines≈6
  - reason: 规模≈6 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:394; kind=function; refs≈4; lines≈6
- **SIMPLIFY** `_canonical_base_url` [function] simp=部分 refs≈4 lines≈12
  - reason: 规模≈12 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:318; kind=function; refs≈4; lines≈12
- **SIMPLIFY** `_hint_backend` [function] simp=部分 refs≈4 lines≈13
  - reason: 规模≈13 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:379; kind=function; refs≈4; lines≈13
- **SIMPLIFY** `_normalize_score` [function] simp=部分 refs≈4 lines≈14
  - reason: 规模≈14 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:407; kind=function; refs≈4; lines≈14
- **SIMPLIFY** `_extract_cli_flag_value_any` [function] simp=部分 refs≈5 lines≈10
  - reason: 规模≈10 行；引用次数≈5（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:367; kind=function; refs≈5; lines≈10
- **SIMPLIFY** `_replace_flag_value` [function] simp=部分 refs≈5 lines≈18
  - reason: 规模≈18 行；引用次数≈5（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:25; kind=function; refs≈5; lines≈18
- **SIMPLIFY** `_as_major_minor` [function] simp=部分 refs≈5 lines≈19
  - reason: 规模≈19 行；引用次数≈5（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:103; kind=function; refs≈5; lines≈19
- **SIMPLIFY** `_matched_anchors` [function] simp=部分 refs≈5 lines≈20
  - reason: 规模≈20 行；引用次数≈5（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:1057; kind=function; refs≈5; lines≈20
- **SIMPLIFY** `_extract_cli_flag_value` [function] simp=部分 refs≈5 lines≈21
  - reason: 规模≈21 行；引用次数≈5（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:344; kind=function; refs≈5; lines≈21

### `runner/opencode_client.py`

- **SIMPLIFY** `OpenCodeClient._post_message_with_retry` [method] simp=部分 refs≈4 lines≈157
  - reason: 规模≈157 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/opencode_client.py:508; kind=method; refs≈4; lines≈157

### `runner/opencode_tooling.py`

- **SIMPLIFY** `_find_tag_gt` [function] simp=部分 refs≈2 lines≈32
  - reason: 规模≈32 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/opencode_tooling.py:382; kind=function; refs≈2; lines≈32
- **SIMPLIFY** `_extract_json_object` [function] simp=部分 refs≈2 lines≈35
  - reason: 规模≈35 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/opencode_tooling.py:345; kind=function; refs≈2; lines≈35
- **SIMPLIFY** `_sanitized_env` [function] simp=部分 refs≈3 lines≈24
  - reason: 规模≈24 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/opencode_tooling.py:695; kind=function; refs≈3; lines≈24
- **SIMPLIFY** `_is_env_like` [function] simp=部分 refs≈4 lines≈20
  - reason: 规模≈20 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/opencode_tooling.py:656; kind=function; refs≈4; lines≈20
- **SIMPLIFY** `_decode_attr_value` [function] simp=部分 refs≈4 lines≈26
  - reason: 规模≈26 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/opencode_tooling.py:88; kind=function; refs≈4; lines≈26
- **SIMPLIFY** `_try_json` [function] simp=部分 refs≈6 lines≈128
  - reason: 规模≈128 行；引用次数≈6（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/opencode_tooling.py:219; kind=function; refs≈6; lines≈128
- **SIMPLIFY** `execute_tool_calls` [function] simp=部分 refs≈8 lines≈286
  - reason: 规模≈286 行；引用次数≈8（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/opencode_tooling.py:902; kind=function; refs≈8; lines≈286

### `runner/prompts.py`

- **SIMPLIFY** `make_scaffold_contract_prompt` [function] simp=部分 refs≈10 lines≈239
  - reason: 规模≈239 行；引用次数≈10（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/prompts.py:105; kind=function; refs≈10; lines≈239

### `runner/repo_resolver.py`

- **SIMPLIFY** `_download_hf_dataset_snapshot` [function] simp=部分 refs≈2 lines≈137
  - reason: 规模≈137 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/repo_resolver.py:130; kind=function; refs≈2; lines≈137

