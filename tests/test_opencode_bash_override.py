from runner.opencode_client import select_bash_mode


def test_select_bash_mode_scaffold_contract_overrides_default():
    assert (
        select_bash_mode(purpose="scaffold_contract", default_bash_mode="restricted", scaffold_bash_mode="full") == "full"
    )
    assert (
        select_bash_mode(purpose="SCAFFOLD_CONTRACT", default_bash_mode="restricted", scaffold_bash_mode="full") == "full"
    )


def test_select_bash_mode_non_scaffold_uses_default():
    assert (
        select_bash_mode(purpose="execute_step", default_bash_mode="restricted", scaffold_bash_mode="full")
        == "restricted"
    )
    assert (
        select_bash_mode(purpose="plan_update_attempt_1", default_bash_mode="full", scaffold_bash_mode="restricted")
        == "full"
    )

