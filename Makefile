.PHONY: agent-check
agent-check:  ## Fast checks the Claude Code Stop hook runs
	python3 scripts/agent/check_style.py .
