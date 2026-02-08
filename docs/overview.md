# OpenCode-FSM Runner (Overview)

This repo provides a small, auditable **library** to run and verify a repo-owned contract:

1. Load `pipeline.yml` (or scaffold it via OpenCode if missing)
2. Run selected stages (deploy → rollout → evaluation → benchmark)
3. Validate required JSON artifacts and metrics

The intent is to integrate with agent projects where you need **repeatable benchmark deployment + evaluation** with hard guardrails and artifacts.

## Key design constraints

- **Pipeline is human-owned**: the runner reverts any model edits to the pipeline YAML.
- **Verification is deterministic**: runner executes commands and records artifacts (stdout/stderr, summaries).

## Files the runner uses (in the target repo)

- `pipeline.yml` (optional): verification contract (see `docs/pipeline_spec.md`)
- `.aider_fsm/bootstrap.yml` (optional): repo-owned environment setup (see `docs/bootstrap_spec.md`)
- `.aider_fsm/runtime_env.json`: runtime connection info for rollout/evaluation scripts (optional)
- `.aider_fsm/artifacts/<run_id>/...`: verification and actions artifacts

## Related docs

- `docs/pipeline_spec.md`
- `docs/bootstrap_spec.md`
- `docs/metrics_schema.md`
- `docs/env_api.md` (library API: `setup()` → `rollout()` → `evaluate()`)
- `docs/verification.md` (smoke/full-lite verification commands + evidence)
