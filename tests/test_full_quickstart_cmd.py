from fsm_runner import _default_full_quickstart_test_cmd


def test_full_quickstart_cmd_contains_expected_checks():
    cmd = _default_full_quickstart_test_cmd()
    assert "kubectl wait" in cmd
    assert "aiopslab/config.yml" in cmd
    assert ".venv_aiopslab/bin/python" in cmd
    assert "Orchestrator" in cmd

