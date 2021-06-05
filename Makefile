run:
	uvicorn main:app --reload

format:
	black -l 120 main.py galahad/ tests/

	isort main.py  galahad/ tests/

test:
	python -m spacy download en_core_web_sm
	python -m pytest tests/
