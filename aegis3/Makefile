.PHONY: help install dev test lint fmt typecheck api orchestrator cli db-up db-down db-migrate workers-build clean

PY ?= python3.12
VENV ?= .venv
ACT = . $(VENV)/bin/activate

help:
	@echo "Targets:"
	@echo "  install         create venv and install package + dev deps"
	@echo "  dev             install in editable mode"
	@echo "  test            run pytest"
	@echo "  lint            ruff check"
	@echo "  fmt             ruff format"
	@echo "  typecheck       mypy"
	@echo "  api             run aegis-api on 127.0.0.1:8787"
	@echo "  orchestrator    run aegis-orchestrator worker"
	@echo "  cli             run aegis CLI (pass ARGS=...)"
	@echo "  db-up           start postgres + redis via docker compose"
	@echo "  db-down         stop postgres + redis"
	@echo "  db-migrate      apply SQL migrations"
	@echo "  workers-build   build all worker images"
	@echo "  clean           remove venv and caches"

$(VENV)/bin/python:
	$(PY) -m venv $(VENV)
	$(ACT) && pip install -U pip wheel

install: $(VENV)/bin/python
	$(ACT) && pip install -e ".[dev]"

dev: install

test:
	$(ACT) && pytest -q

lint:
	$(ACT) && ruff check .

fmt:
	$(ACT) && ruff format .

typecheck:
	$(ACT) && mypy src

api:
	$(ACT) && aegis-api

orchestrator:
	$(ACT) && aegis-orchestrator

cli:
	$(ACT) && aegis $(ARGS)

db-up:
	docker compose -f deploy/docker-compose.yml up -d postgres redis

db-down:
	docker compose -f deploy/docker-compose.yml down

db-migrate:
	$(ACT) && python -m aegis3.api.db migrate

workers-build:
	for t in slither foundry echidna medusa mythril halmos; do \
	  docker build -t aegis3/$$t:dev workers/$$t || exit 1 ; \
	done

clean:
	rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
