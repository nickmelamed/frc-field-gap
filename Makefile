.PHONY: help setup lint typecheck test check requirements agent-check download

help:  ## List targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

setup:  ## Install the locked environment and the git hooks
	uv sync --locked
	uv run pre-commit install

lint:  ## Lint and check formatting
	uv run ruff check .
	uv run ruff format --check .

typecheck:  ## Type-check src/ with mypy --strict
	uv run mypy

test:  ## Run unit tests with coverage (integration tests skipped)
	uv run pytest

check: lint typecheck test  ## Lint, typecheck, and test

requirements:  ## Re-export requirements.txt for Colab from uv.lock
	uv export --locked --format requirements-txt --no-dev --no-emit-project --no-hashes -o requirements.txt

agent-check: check  ## Fast checks the Claude Code Stop hook runs
	python3 scripts/agent/check_style.py .
	python3 scripts/agent/check_numbers.py README.md docs/DATASETS.md docs/EVALUATION.md docs/MODEL_CARD.md docs/DEPLOYMENT.md --sources reports

download:  ## Download pinned datasets to data/raw/ (ARGS=--resolve to look them up)
	uv run frc-download $(ARGS)
