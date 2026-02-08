from __future__ import annotations

from pathlib import Path

from runner.prompts import make_scaffold_contract_prompt, make_scaffold_contract_retry_prompt


def test_scaffold_contract_prompt_mentions_trained_model_dir_and_runtime_env():
    # 作用：pytest 测试用例：验证行为契约
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈7 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_prompts_scaffold_contract.py:9；类型=function；引用≈1；规模≈7行
    p = make_scaffold_contract_prompt(Path("/tmp/repo"), pipeline_rel="pipeline.yml", require_metrics=True)
    assert "AIDER_RUNTIME_ENV_PATH" in p
    assert "AIDER_TRAINED_MODEL_DIR" in p
    assert "AIDER_LLM_KIND" in p
    assert "AIDER_LLM_MODEL" in p
    assert "model_dir" in p


def test_scaffold_contract_prompt_includes_doc_command_hints_when_provided():
    # 作用：pytest 测试用例：验证行为契约
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈9 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_prompts_scaffold_contract.py:18；类型=function；引用≈1；规模≈9行
    p = make_scaffold_contract_prompt(
        Path("/tmp/repo"),
        pipeline_rel="pipeline.yml",
        require_metrics=True,
        command_hints=["python -m benchtool.evaluate --dataset demo --model openai"],
    )
    assert "[CANDIDATE_COMMAND_HINTS]" in p
    assert "benchtool.evaluate" in p


def test_scaffold_contract_prompt_forbids_fake_tool_transcripts():
    # 作用：pytest 测试用例：验证行为契约
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈4 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_prompts_scaffold_contract.py:29；类型=function；引用≈1；规模≈4行
    p = make_scaffold_contract_prompt(Path("/tmp/repo"), pipeline_rel="pipeline.yml", require_metrics=True)
    assert "Do NOT fabricate `<tool_result>` blocks" in p
    assert "Do NOT print pseudo tool snippets as plain text" in p


def test_scaffold_contract_prompt_mentions_opencode_xml_tool_formats():
    # 作用：pytest 测试用例：验证行为契约
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈5 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_prompts_scaffold_contract.py:35；类型=function；引用≈1；规模≈5行
    p = make_scaffold_contract_prompt(Path("/tmp/repo"), pipeline_rel="pipeline.yml", require_metrics=True)
    assert '<read filePath="PATH" />' in p
    assert '<write filePath="PATH">' in p
    assert '<bash command="..." description="..." />' in p


def test_scaffold_contract_retry_prompt_forbids_pseudo_tool_syntax():
    # 作用：pytest 测试用例：验证行为契约
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈11 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_prompts_scaffold_contract.py:42；类型=function；引用≈1；规模≈11行
    p = make_scaffold_contract_retry_prompt(
        Path("/tmp/repo"),
        pipeline_rel="pipeline.yml",
        require_metrics=True,
        attempt=2,
        max_attempts=3,
        previous_failure="missing_pipeline_yml",
    )
    assert "Do NOT output pseudo tool syntax" in p
    assert "emit fake `<tool_result>` blocks" in p
