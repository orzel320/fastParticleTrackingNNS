.PHONY: all clean install generate test benchmark

all: install clean generate test benchmark

all2: clean generate test benchmark

install:
	pip install -e .[dev]

clean:
	rm -rf data/
	rm -f scaling.png
	rm -rf .pytest_cache/
	rm -rf src/hep_tracking/__pycache__/
	rm -rf tests/__pycache__/

generate:
	python src/hep_tracking/data.py

test:
	pytest tests/

benchmark: generate
	python src/hep_tracking/benchmark.py