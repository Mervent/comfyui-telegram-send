APP_NAME := comfyui_telegram

venv:
	uv venv .venv
	uv pip install -r pyproject.toml

lint:
	ruff check ${APP_NAME}
	ty check ${APP_NAME}

format:
	ruff format ${APP_NAME}

test:
	pytest -v ${APP_NAME}
