# Contributing to RevNet-Zero

We welcome contributions from the community! This guide will help you get started.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/revnet-zero.git`
3. Install development dependencies: `pip install -e ".[dev]"`
4. Create a feature branch: `git checkout -b feature/your-feature`

## Development Setup

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
python -m pytest tests/

# Run linting
python -m flake8 revnet_zero/
python -m black revnet_zero/ tests/

# Run type checking
python -m mypy revnet_zero/
```

## Contributing Guidelines

### Code Quality
- Follow PEP 8 style guidelines
- Add type hints for all public APIs
- Write comprehensive docstrings
- Maintain >90% test coverage
- Use descriptive variable and function names

### Testing
- Write tests for all new functionality
- Include both unit and integration tests
- Verify gradient correctness for reversible layers
- Test memory usage and performance characteristics

### Documentation
- Update API documentation for new features
- Add examples for complex functionality
- Update README.md if needed
- Include theoretical background for algorithms

### Pull Request Process

1. Update tests and documentation
2. Ensure all tests pass
3. Run linting and type checking
4. Submit pull request with clear description
5. Address review feedback promptly

## Bug Reports

When filing bug reports, please include:
- Python version and environment details
- Minimal reproducible example
- Expected vs actual behavior
- Relevant error messages and stack traces

## Feature Requests

For feature requests, please:
- Describe the use case and motivation
- Provide example usage if possible
- Consider implementation complexity
- Discuss alternatives you've considered

## Research Contributions

We welcome research contributions including:
- Theoretical analysis of reversible computing
- Novel coupling function designs
- Performance optimization techniques
- Empirical studies on long-context training

## Community

- Join our Discord for discussions
- Follow our Twitter for updates
- Attend virtual office hours
- Participate in community calls