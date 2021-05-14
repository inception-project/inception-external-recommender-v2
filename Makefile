run:
	uvicorn main:app --reload

format:
	black -l 120 main.py galahad/

	isort main.py  galahad/
