# Contributing to iAODE

Thank you for your interest in contributing to iAODE! This document provides guidelines for contributions.

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/PeterPonyu/iAODE.git
cd iAODE
```

2. Create a development environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in development mode:
```bash
pip install -e ".[dev]"
```

4. Install pre-commit hooks (optional but recommended):
```bash
pip install pre-commit
pre-commit install
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for all public functions/classes
- Format code with Black (line length: 100)
- Run flake8 for linting

```bash
# Format code
black iaode/

# Check linting
flake8 iaode/

# Type checking
mypy iaode/
```

## Testing

We use pytest for testing. Write tests for new features:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=iaode --cov-report=html
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass: `pytest`
6. Format code: `black iaode/`
7. Update documentation if needed
8. Commit changes: `git commit -m 'Add amazing feature'`
9. Push to branch: `git push origin feature/amazing-feature`
10. Open a Pull Request

### PR Guidelines

- Provide a clear description of the changes
- Reference any related issues
- Include examples if adding new features
- Ensure CI passes
- Request review from maintainers

## Adding New Features

### Adding a New Encoder Type

1. Implement the encoder in `iaode/module.py`:
```python
class NewEncoder(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Your implementation
    
    def forward(self, x):
        # Your implementation
        return q_z, q_m, q_s
```

2. Update the encoder factory in `module.py`:
```python
if encoder_type == "new_encoder":
    self.encoder = NewEncoder(...)
```

3. Add documentation and tests
4. Update README with usage example

### Adding a New Loss Function

1. Implement loss calculation in `iaode/model.py` or `iaode/mixin.py`
2. Add loss weight parameter to `iVAE.__init__()`
3. Update total loss calculation in `iVAE.update()`
4. Document the new loss term
5. Add tests

### Adding Evaluation Metrics

1. Implement metric in `iaode/DRE.py` or `iaode/LSE.py`
2. Add to comprehensive evaluation function
3. Document metric interpretation
4. Add examples

## Documentation

- Update README.md for major features
- Add examples in `examples/` directory
- Update QUICKSTART.md if relevant
- Write clear docstrings with Args, Returns, Examples

### Docstring Format

```python
def my_function(param1: int, param2: str) -> bool:
    """
    Brief description of function.
    
    Longer description with more details if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Example:
        >>> my_function(1, "test")
        True
    """
    pass
```

## Reporting Bugs

Use GitHub Issues with the following information:

- Python version
- iAODE version
- Operating system
- Minimal reproducible example
- Error message and stack trace
- Expected behavior

## Feature Requests

Open a GitHub Issue with:

- Clear description of the feature
- Use case / motivation
- Possible implementation approach
- Willingness to contribute

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Help others learn

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

- Open a GitHub Discussion for questions
- Check existing issues for similar questions
- Tag maintainers for urgent matters

Thank you for contributing to iAODE!
