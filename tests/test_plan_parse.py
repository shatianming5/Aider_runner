from runner.plan_format import parse_backlog_open_count, parse_next_step, parse_plan


def _plan(text: str) -> str:
    """中文说明：
    - 含义：测试用 helper：生成一个最小但结构完整的 PLAN.md 文本。
    - 内容：填充 Goal/Acceptance/Next/Backlog/Done/Notes，并把 `text` 注入到 Next 区块。
    - 可简略：是（只是构造样本；也可直接在各测试内写字符串）。
    """
    return (
        "# PLAN\n\n"
        "## Goal\n"
        "- g\n\n"
        "## Acceptance\n"
        "- [ ] TEST_CMD passes: `pytest -q`\n\n"
        "## Next (exactly ONE item)\n"
        f"{text}\n"
        "## Backlog\n"
        "- [ ] (STEP_ID=002) b\n\n"
        "## Done\n"
        "- [x] (STEP_ID=000) d\n\n"
        "## Notes\n"
        "- \n"
    )


def test_parse_next_step_ok():
    """中文说明：
    - 含义：验证 `parse_next_step` 在 Next 区块恰好 1 项且格式正确时返回 step。
    - 内容：构造一个 Next 含 `(STEP_ID=001)` 的 PLAN，断言解析出的 id/text 正确且 err 为 None。
    - 可简略：可能（可与其它 next 相关测试合并参数化；当前粒度清晰）。
    """
    step, err = parse_next_step(_plan("- [ ] (STEP_ID=001) do x\n"))
    assert err is None
    assert step == {"id": "001", "text": "do x"}


def test_parse_next_step_missing_section():
    """中文说明：
    - 含义：验证缺失 Next 区块时会返回明确错误码。
    - 内容：输入最小文本 `# PLAN`，断言 step 为 None 且 err 为 missing_next_section。
    - 可简略：是（典型负例）。
    """
    step, err = parse_next_step("# PLAN\n")
    assert step is None
    assert err == "missing_next_section"


def test_parse_next_step_multiple_items():
    """中文说明：
    - 含义：验证 Next 区块出现多条待办时会报错（要求 exactly ONE item）。
    - 内容：构造 Next 含两条 item，断言 step=None 且 err=next_count_not_one。
    - 可简略：是（典型负例）。
    """
    step, err = parse_next_step(_plan("- [ ] (STEP_ID=001) a\n- [ ] (STEP_ID=003) b\n"))
    assert step is None
    assert err == "next_count_not_one"


def test_parse_next_step_checked_is_error():
    """中文说明：
    - 含义：验证 Next 条目被勾选（[x]）时视为错误。
    - 内容：构造 Next 为 `- [x] ...`，断言 err=next_is_checked。
    - 可简略：是（典型负例）。
    """
    step, err = parse_next_step(_plan("- [x] (STEP_ID=001) done\n"))
    assert step is None
    assert err == "next_is_checked"


def test_parse_next_step_bad_line_is_error():
    """中文说明：
    - 含义：验证 Next 条目格式不符合 `(STEP_ID=...)` 规范时会报错。
    - 内容：构造一个缺少括号/格式错误的行，断言 err=bad_next_line。
    - 可简略：是（典型负例）。
    """
    step, err = parse_next_step(_plan("- [ ] STEP_ID=001 bad\n"))
    assert step is None
    assert err == "bad_next_line"


def test_backlog_open_count_ok():
    """中文说明：
    - 含义：验证 `parse_backlog_open_count` 能正确统计 Backlog 中未完成项数量。
    - 内容：构造 Backlog 里 1 条未完成项，断言 open_count=1 且 err 为 None。
    - 可简略：可能（可补充更多组合；当前覆盖主路径）。
    """
    plan = _plan("- [ ] (STEP_ID=001) x\n") + "\n"
    open_count, err = parse_backlog_open_count(plan)
    assert err is None
    assert open_count == 1


def test_duplicate_step_id_is_error():
    """中文说明：
    - 含义：验证 `parse_plan` 会检查 Next/Backlog/Done 中的 STEP_ID 不可重复。
    - 内容：构造 Next 与 Backlog 都包含 STEP_ID=001，断言 parsed.errors 包含 duplicate_step_id。
    - 可简略：否（是计划格式的关键约束，建议保留）。
    """
    plan = (
        "# PLAN\n\n"
        "## Goal\n- g\n\n"
        "## Acceptance\n- [ ] TEST_CMD passes: `pytest -q`\n\n"
        "## Next (exactly ONE item)\n"
        "- [ ] (STEP_ID=001) a\n\n"
        "## Backlog\n"
        "- [ ] (STEP_ID=001) dup\n\n"
        "## Done\n"
        "- [x] (STEP_ID=000) d\n\n"
        "## Notes\n- \n"
    )
    parsed = parse_plan(plan)
    assert "duplicate_step_id" in parsed["errors"]
