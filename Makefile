.DEFAULT_GOAL := help
.PHONY: help setup venv env doctor install install-engine install-metal install-dev \
        pull serve up demo test smoke lint fmt clean

# --- interpreter -------------------------------------------------------
# Recursive (not :=) so that `make setup` picks up the venv it just created:
# the wildcard is re-evaluated every time $(PY) is used.
VENV := .venv
PY = $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)

# --- what to install and serve ----------------------------------------
EXTRAS ?= engine,dev
MODEL  ?= hf:Qwen/Qwen2.5-1.5B-Instruct-GGUF:q4_k_m
REF    ?= qwen2.5-1.5b-instruct-gguf:q4_k_m

# --- platform ---------------------------------------------------------
# Metal is the difference between ~50 tok/s and ~4 tok/s on an Apple Silicon
# machine, so it is detected rather than remembered.
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_S)-$(UNAME_M),Darwin-arm64)
  METAL := CMAKE_ARGS="-DGGML_METAL=on"
  PLATFORM := macOS $(UNAME_M) — building llama.cpp with Metal
else
  METAL :=
  PLATFORM := $(UNAME_S) $(UNAME_M) — CPU build
endif

help: ## Show this help
	@echo "shilp — pull, reshape, serve. Targets:"
	@echo ""
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  Platform detected: $(PLATFORM)"
	@echo "  Model default:     $(MODEL)  →  $(REF)"

setup: venv env ## End-to-end install: venv, dependencies, .env, doctor
	@echo "→ $(PLATFORM)"
	PIP_USER=0 $(PY) -m pip install --quiet --upgrade pip
	PIP_USER=0 $(METAL) $(PY) -m pip install -e ".[$(EXTRAS)]"
	@echo ""
	@$(PY) -m shilp.cli doctor
	@echo ""
	@echo "Next:  make test     # 78 tests, no model needed"
	@echo "       make pull     # ~1 GB model from the Hub"
	@echo "       make serve    # http://127.0.0.1:8000"
	@echo "  or:  make up       # pull + serve in one step"

venv: ## Create .venv if it is missing
	@test -d $(VENV) || $(PY) -m venv $(VENV)

env: ## Create .env from .env.example if it is missing
	@test -f .env || (cp .env.example .env && echo "→ wrote .env from .env.example")

doctor: ## Report what this machine has
	@$(PY) -m shilp.cli doctor

pull: ## Download the default model unless it is already stored
	@$(PY) -m shilp.cli ls 2>/dev/null | grep -q '^$(REF) ' \
	  && echo "→ $(REF) already stored" \
	  || $(PY) -m shilp.cli pull $(MODEL)

serve: ## Serve the default model on :8000
	$(PY) -m shilp.cli serve --model $(REF)

up: setup pull serve ## Install, download the model, and serve — one command

install: ## Install the package only (no engine, no dev tools)
	PIP_USER=0 $(PY) -m pip install -e .

install-engine: ## Install with the llama.cpp engine (CPU build)
	PIP_USER=0 $(PY) -m pip install -e ".[engine]"

install-metal: ## Install with the llama.cpp engine built for Metal
	PIP_USER=0 CMAKE_ARGS="-DGGML_METAL=on" $(PY) -m pip install -e ".[engine]"

install-dev: ## Install with engine and dev tools
	PIP_USER=0 $(PY) -m pip install -e ".[engine,dev]"

test: ## Run the unit suite (no model, no network)
	$(PY) -m pytest -q

smoke: ## Boot a real server and exercise every endpoint over HTTP
	$(PY) scripts/smoke_test.py

demo: ## Serve with the fake engine — the API with no model behind it
	SHILP_ENGINE=fake $(PY) -m shilp.cli serve --port 8000

lint: ## Ruff, if it is installed
	@$(PY) -m ruff check src tests scripts 2>/dev/null || echo "ruff not installed (pip install ruff)"

fmt: ## Ruff format, if it is installed
	@$(PY) -m ruff format src tests scripts 2>/dev/null || echo "ruff not installed (pip install ruff)"

clean: ## Remove build artefacts and the venv
	rm -rf build dist *.egg-info .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	@echo "left .venv and .env alone; remove them by hand if you want a clean slate"
