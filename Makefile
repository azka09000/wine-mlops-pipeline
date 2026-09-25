PYTHON ?= python

.PHONY: install lint test train clean

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

lint:
	$(PYTHON) -m flake8 src/ tests/ --max-line-length=100

test:
	$(PYTHON) -m pytest tests/ -v

train:
	$(PYTHON) -m src.train

clean:
	find . -type f -name "*.pyc" -not -path "./.venv/*" -delete
	find . -type d -name "__pycache__" -not -path "./.venv/*" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .coverage