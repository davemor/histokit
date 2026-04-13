.PHONY: fmt tests

fmt:
	uv run ruff check --fix
	uv run ruff format

tests:
	uv run pytest tests/