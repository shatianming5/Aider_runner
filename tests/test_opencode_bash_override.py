from runner.opencode_client import select_bash_mode


def test_select_bash_mode_scaffold_contract_overrides_default():
    """中文说明：
    - 含义：验证 scaffold_contract 场景下优先使用 scaffold_bash_mode 覆盖默认值。
    - 内容：分别用不同大小写的 purpose 调用 `select_bash_mode`，断言返回 scaffold_bash_mode。
    - 可简略：可能（可用参数化减少重复；但当前可读性较强）。
    """
    assert (
        select_bash_mode(purpose="scaffold_contract", default_bash_mode="restricted", scaffold_bash_mode="full") == "full"
    )
    assert (
        select_bash_mode(purpose="SCAFFOLD_CONTRACT", default_bash_mode="restricted", scaffold_bash_mode="full") == "full"
    )


def test_select_bash_mode_non_scaffold_uses_default():
    """中文说明：
    - 含义：验证非 scaffold_contract 场景使用 default_bash_mode。
    - 内容：分别测试 execute_step 与 plan_update_attempt_1 两种 purpose，断言返回 default_bash_mode。
    - 可简略：可能（可参数化；但覆盖两条主路径已足够）。
    """
    assert (
        select_bash_mode(purpose="execute_step", default_bash_mode="restricted", scaffold_bash_mode="full")
        == "restricted"
    )
    assert (
        select_bash_mode(purpose="plan_update_attempt_1", default_bash_mode="full", scaffold_bash_mode="restricted")
        == "full"
    )
