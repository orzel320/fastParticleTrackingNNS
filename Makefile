.PHONY: all clean install data_pipeline generate candidates test
	
all: install clean data_pipeline test

install:
	python -m pip install -e .[dev]

clean:
	rm -rf data/
	rm -rf .pytest_cache/
	rm -rf .ipynb_checkpoints/
	rm -rf src/hep_tracking/__pycache__/
	rm -rf tests/__pycache__/
	rm -rf *.egg-info

data_pipeline: generate candidates

generate:
	python src/hep_tracking/data.py

candidates:
	python src/hep_tracking/generate_candidates.py

test:
	pytest tests/