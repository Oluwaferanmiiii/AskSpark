.PHONY: help install install-dev test lint format clean run

# Default target
help:
	@echo "Available commands:"
	@echo "  install      Install package dependencies"
	@echo "  install-dev   Install development dependencies"
	@echo "  test         Run tests"
	@echo "  lint         Run linting"
	@echo "  format       Format code"
	@echo "  clean        Clean temporary files"
	@echo "  run          Run the application"

# Install dependencies
install:
	pip install -r requirements.txt
	pip install -e .

# Install development dependencies
install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"

# Run tests
test:
	pytest tests/ -v --cov=src/askspark --cov-report=html --cov-report=term

# Run linting
lint:
	flake8 src/ tests/
	mypy src/

# Format code
format:
	black src/ tests/
	isort src/ tests/

# Clean temporary files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# Run the application
run:
	python app.py
