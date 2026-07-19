.PHONY: run format format-check lint typecheck security test check

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

test:
	pytest

check: format-check lint typecheck security test
