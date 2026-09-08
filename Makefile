all: run

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	rm -rf .mypy_cache data/output
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

.PHONY: all install run debug clean lint lint-strict
