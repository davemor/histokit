.PHONY: fmt tests build publish

fmt:
	uv run ruff check --fix
	uv run ruff format

tests:
	uv run pytest tests/

build:
	uv build
