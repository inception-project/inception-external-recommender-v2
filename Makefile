run:
	uvicorn main:app --reload

format:
	black -l 120 main.py setup.py galahad/ tests/ scripts/

	isort main.py setup.py galahad/ tests/ scripts/

test: get_test_dependencies
	python -m pytest tests/

inception_test: get_test_dependencies
	python scripts/inception_integration_test.py

get_test_dependencies:
	python -m spacy download en_core_web_sm

build:
	python -m build