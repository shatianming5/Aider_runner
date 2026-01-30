.PHONY: bootstrap ci smoke run

bootstrap:
	bash scripts/bootstrap.sh

smoke:
	bash scripts/smoke_test.sh

ci:
	bash scripts/ci.sh

run:
	bash scripts/run_local.sh
