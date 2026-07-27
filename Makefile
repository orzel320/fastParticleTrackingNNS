.PHONY: all clean install install-gpu data_pipeline generate candidates test format lint notebook

all: install clean data_pipeline test

install:
	python -m pip install -e .[dev]

install-gpu:
	python -m pip install -e .[dev,gpu]

clean:
	rm -rf data/
	rm -f *.pdf

data_pipeline: generate candidates

generate:
	python src/hep_tracking/data.py

candidates:
	python src/hep_tracking/generate_candidates.py

test:
	pytest tests/

format:
	black src tests
	isort src tests

lint:
	black --check src tests
	isort --check-only src tests
