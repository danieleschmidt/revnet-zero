# RevNet-Zero Makefile

.PHONY: help install install-dev test lint format type-check clean build docs

# Default target
help:
	@echo "Available commands:"
	@echo "  install      Install package in development mode"
	@echo "  install-dev  Install package with development dependencies"
	@echo "  test         Run tests"
	@echo "  lint         Run linting"
	@echo "  format       Format code"
	@echo "  type-check   Run type checking"
	@echo "  clean        Clean build artifacts"
	@echo "  build        Build package"
	@echo "  docs         Build documentation"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# Testing
test:
	python -m pytest tests/ -v --cov=revnet_zero --cov-report=html --cov-report=term

test-fast:
	python -m pytest tests/ -v -x

test-memory:
	python -m pytest tests/test_memory_scheduler.py -v

# Code quality
lint:
	flake8 revnet_zero/ tests/
	black --check revnet_zero/ tests/
	isort --check-only revnet_zero/ tests/

format:
	black revnet_zero/ tests/
	isort revnet_zero/ tests/

type-check:
	mypy revnet_zero/

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Building
build: clean
	python -m build

# Documentation
docs:
	@echo "Documentation build not yet implemented"

# Development shortcuts
dev-setup: install-dev
	@echo "Development environment ready!"

check: lint type-check test-fast
	@echo "All checks passed!"

# Benchmarking
benchmark:
	python -m revnet_zero.cli.benchmark --help

# Memory profiling
profile-memory:
	python scripts/profile_memory.py

# Model conversion
convert-model:
	python -m revnet_zero.cli.convert --help