.PHONY: install install-engine install-engine-metal install-dev test smoke demo lint clean fmt

PY ?= python

install:
	$(PY) -m pip install -e .

# CPU build. On Apple Silicon use install-engine-metal instead, or you will get
# CPU-only inference at roughly a tenth of the speed.
install-engine:
	$(PY) -m pip install -e ".[engine]"

install-engine-metal:
	CMAKE_ARGS="-DGGML_METAL=on" $(PY) -m pip install -e ".[engine]"

install-dev:
	$(PY) -m pip install -e ".[dev,engine]"

# 64 unit tests. No model, no network, a few seconds.
test:
	$(PY) -m pytest -q

# Boots a real uvicorn server and exercises every endpoint over HTTP.
smoke:
	$(PY) scripts/smoke_test.py

# The API with no model behind it — for UI work and demos.
demo:
	SHILP_ENGINE=fake $(PY) -m shilp.cli serve --port 8000

lint:
	ruff check src tests scripts || true

fmt:
	ruff format src tests scripts || true

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
