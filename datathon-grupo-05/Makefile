.PHONY: install train serve lint test

install:
	python -m pip install -e .[dev]

train:
	python src/models/train.py

serve:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

lint:
	ruff check src tests

test:
	pytest tests --cov=src --cov-fail-under=60
