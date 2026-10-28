# MAgent - Mr. Agent, Powerful.
# Zero-dependency Python agent framework built for Cortex LLMHoster.
#
# Every target below works with a stock CPython 3.11+ interpreter.
# Nothing here requires pip installs, network access, or a running Cortex.

PYTHON ?= python3
SRC    := src
TESTS  := tests

.DEFAULT_GOAL := help

.PHONY: help install test unit smoke smoke-tools tools lint type fmt check verify count
.PHONY: demo examples doctor run serve bench eval all clean dist improvements verify-quick

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install MAgent into the current interpreter (editable, no deps)
	$(PYTHON) -m pip install -e . --no-deps

test: ## Run the full unittest suite (stdlib only, no pytest needed)
	PYTHONPATH=$(SRC) $(PYTHON) -m unittest discover -s $(TESTS) -t . -v

unit: ## Run the fast core tests only
	PYTHONPATH=$(SRC) $(PYTHON) -m unittest discover -s $(TESTS) -t . -p 'test_core*.py'

smoke: ## Run the end-to-end offline smoke scenario
	PYTHONPATH=$(SRC) $(PYTHON) scripts/smoke.py

smoke-tools: ## Exercise every built-in tool against a throwaway workspace
	PYTHONPATH=$(SRC) $(PYTHON) scripts/smoke_tools.py

tools: ## List every built-in tool by category
	PYTHONPATH=$(SRC) $(PYTHON) -m magent tools list

improvements: ## Regenerate improvements.json and IMPROVEMENTS.md from the source tree
	$(PYTHON) scripts/build_improvements.py

verify: ## Prove every numbered improvement points at real code
	$(PYTHON) scripts/verify_improvements.py --strict

verify-quick: ## Verify the manifest references without regenerating
	$(PYTHON) scripts/verify_improvements.py

count: ## Print how many improvements are registered
	$(PYTHON) scripts/count_improvements.py

lint: ## Style check with ruff if it happens to be installed
	@$(PYTHON) -m ruff check $(SRC) $(TESTS) examples scripts || \
		echo "ruff not installed - skipped (MAgent has no mandatory dev deps)"

type: ## Type check with mypy if it happens to be installed
	@$(PYTHON) -m mypy $(SRC)/magent || \
		echo "mypy not installed - skipped"

fmt: ## Auto-format with ruff if it happens to be installed
	@$(PYTHON) -m ruff format $(SRC) $(TESTS) examples scripts || echo "ruff not installed"

check: test smoke smoke-tools verify ## Tests + smoke runs + improvement verification

demo: ## Offline demo against the built-in Echo backend
	PYTHONPATH=$(SRC) $(PYTHON) examples/01_hello_agent.py

examples: ## Run every example offline
	@for file in examples/0*.py; do echo "== $$file"; PYTHONPATH=$(SRC) $(PYTHON) $$file >/dev/null || exit 1; done; echo "all examples ran"

eval: ## Run the sample evaluation dataset and gate on the pass rate
	PYTHONPATH=$(SRC) $(PYTHON) -m magent eval --offline --threshold 0.5 --report-format text

doctor: ## Diagnose the local Cortex LLMHoster connection
	PYTHONPATH=$(SRC) $(PYTHON) -m magent doctor

run: ## One-shot prompt: make run P="explain closures"
	PYTHONPATH=$(SRC) $(PYTHON) -m magent run "$(P)"

serve: ## Start the local MAgent HTTP server + chat UI
	PYTHONPATH=$(SRC) $(PYTHON) -m magent serve --host 0.0.0.0 --port 8700

bench: ## Benchmark the agent loop offline
	PYTHONPATH=$(SRC) $(PYTHON) -m magent bench

dist: ## Build wheel + sdist (needs hatchling)
	$(PYTHON) -m pip install --quiet build && $(PYTHON) -m build

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov

all: clean test verify smoke smoke-tools ## Full clean rebuild + all checks
