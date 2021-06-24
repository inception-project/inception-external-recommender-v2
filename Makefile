run:
	uvicorn main:app --reload

format:
	black -l 120 main.py setup.py galahad/ tests/ scripts/

	isort main.py setup.py galahad/ tests/ scripts/

test: get_test_dependencies
	python -m pytest tests/

get_test_dependencies:
	python -m spacy download en_core_web_sm