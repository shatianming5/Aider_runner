"""OpenCode-FSM runner core package.

Library-first API for running a repo-owned verification contract (`pipeline.yml` + `.aider_fsm/`).
Public usage is intentionally minimal: `setup()` -> `sess.rollout(llm=...)` -> `sess.evaluate()`.
"""
