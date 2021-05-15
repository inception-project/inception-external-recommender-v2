run:
	uvicorn main:app --reload

format:
	black -l 120 main.py galahad/ tests/

	isort main.py  galahad/ tests/
