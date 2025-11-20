# Contributing to iAODE

Thank you for your interest in contributing to iAODE!

## Quick Setup

```bash
git clone https://github.com/PeterPonyu/iAODE.git
cd iAODE
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

## Code Standards

- Follow PEP 8 guidelines
- Use type hints and docstrings for public functions
- Format with Black (line length: 100)
- Test with pytest before submitting

```bash
black iaode/        # Format code
pytest              # Run tests
```

## Contributing Workflow

1. **Fork** the repository
2. **Create** feature branch: `git checkout -b feature/name`
3. **Develop** your changes with tests
4. **Format** code: `black iaode/`
5. **Test**: `pytest`
6. **Commit**: `git commit -m 'Add feature'`
7. **Push**: `git push origin feature/name`
8. **Submit** Pull Request with clear description

## Adding Features

### New Encoder Type

Add to `iaode/module.py`:
```python
class NewEncoder(nn.Module):
    def forward(self, x):
        return q_z, q_m, q_s
```

Update encoder factory and add tests.

### New Loss Function

Implement in `iaode/model.py`, update `iVAE.update()`, document and test.

### Evaluation Metrics

Add to `iaode/DRE.py` or `iaode/LSE.py`, document interpretation, provide examples.

## Documentation

- Update README.md for major features
- Add examples in `examples/` directory
- Use clear docstrings with Args, Returns, Examples

## Reporting Issues

Include: Python version, iAODE version, OS, minimal example, error trace, expected behavior.

## Questions?

Open a GitHub Discussion or check existing issues.

## License

Contributions are licensed under the MIT License.

Thank you for contributing!
