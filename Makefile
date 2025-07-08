venv:
	uv venv .venv
	uv pip install -r pyproject.toml

lint:
	ruff check ./comfyu_telegram ./tests
	ty check ./comfyu_telegram ./tests

format:
	ruff format ./comfyu_telegram ./tests

test:
	pytest -v ./tests/
