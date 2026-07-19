.PHONY: run format format-check lint typecheck security dependencies test evaluation check

run:
	python -m linked_ai_content_pipeline --config configs/pipeline.toml --fixture fixtures/materials.json

format:
	ruff format .
	ruff check --fix .

format-check:
	ruff format --check .

lint:
	ruff check .

typecheck:
	mypy src tests

security:
	bandit -q -r src

dependencies:
	pip-audit -r requirements-runtime.lock

test:
	pytest

evaluation:
	python -m linked_ai_content_pipeline.rag_eval
	python -m linked_ai_content_pipeline.llm_eval

check: format-check lint typecheck security dependencies test evaluation
