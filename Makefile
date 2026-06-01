.PHONY: test lint format-check smoke-public secret-scan

test:
	python -m pytest -q

lint:
	python -m ruff check src tests

format-check:
	python -m black --check src tests

smoke-public:
	python -m succession_fragility.cli run-synthetic --output-dir reports

secret-scan:
	python scripts/secret_scan.py .
