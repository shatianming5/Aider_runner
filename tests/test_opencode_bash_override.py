from runner.opencode_client import select_bash_mode


def test_select_bash_mode_scaffold_contract_overrides_default():
    """中文说明：
    - 含义：验证 scaffold_contract 场景下优先使用 scaffold_bash_mode 覆盖默认值。
    - 内容：分别用不同大小写的 purpose 调用 `select_bash_mode`，断言返回 scaffold_bash_mode。
    - 可简略：可能（可用参数化减少重复；但当前可读性较强）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈12 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_opencode_bash_override.py:10；类型=function；引用≈1；规模≈12行
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
    # 作用：中文说明：
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈14 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_opencode_bash_override.py:24；类型=function；引用≈1；规模≈14行
    assert (
        select_bash_mode(purpose="execute_step", default_bash_mode="restricted", scaffold_bash_mode="full")
        == "restricted"
    )
    assert (
        select_bash_mode(purpose="plan_update_attempt_1", default_bash_mode="full", scaffold_bash_mode="restricted")
        == "full"
    )
