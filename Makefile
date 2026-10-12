PY ?= .venv/bin/python
.DEFAULT_GOAL := help

help:            ## Show targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

install:         ## Create .venv and install everything (Python 3.10-3.12)
	python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements-dev.txt && .venv/bin/pip install -e . --no-deps

train:           ## Reproduce training, evaluation, model card and registry entry
	$(PY) -m retentioniq.models.train

samples:         ## Regenerate synthetic sample CSVs
	$(PY) scripts/make_samples.py

seed:            ## Reset and seed the demo database
	$(PY) scripts/seed_demo.py --reset

app:             ## Run the Streamlit app on :8501
	$(PY) -m streamlit run app.py

worker:          ## Run the background worker (jobs, schedules, webhook delivery)
	$(PY) -m retentioniq.jobs.worker

api:             ## Run the REST API on :8000 (docs at /docs)
	$(PY) -m uvicorn api.main:app --reload --port 8000

test:            ## Run the test suite
	$(PY) -m pytest -q

lint:            ## Lint
	$(PY) -m ruff check src api ui app_pages scripts tests app.py

models:          ## List registered models
	$(PY) scripts/registry_cli.py list

docker:          ## Build and run app + API with Docker Compose
	docker compose up --build

.PHONY: help install train samples seed app worker api test lint models docker
