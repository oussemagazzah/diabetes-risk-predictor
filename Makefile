PY ?= python

.PHONY: install train evaluate explain all test lint format app eda

install:
	$(PY) -m pip install -r requirements.txt

train:
	$(PY) -m diabetes_predictor.train

evaluate:
	$(PY) -m diabetes_predictor.evaluate
	$(PY) -m diabetes_predictor.explain

all: train evaluate

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

app:
	$(PY) -m streamlit run app/streamlit_app.py

eda:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
