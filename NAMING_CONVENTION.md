# Repository vs Package Naming

## Important: Different Names by Design

- **GitHub Repository Name**: `iAODE` (mixed case)
- **Python Package Name**: `iaode` (lowercase)

## Why This is Correct

This naming difference is **intentional and follows Python best practices**:

### Python Package Naming (PEP 8)
- Package names should be **lowercase** without hyphens
- PyPI package names are case-insensitive
- Users install with: `pip install iaode`
- Users import with: `import iaode`

### Repository Naming
- GitHub repository names are case-sensitive
- `iAODE` is more readable and matches the project branding
- Clone with: `git clone https://github.com/PeterPonyu/iAODE.git`

## Examples from Popular Projects

Many projects use this pattern:

| GitHub Repository | PyPI Package | Import Name |
|------------------|--------------|-------------|
| `PyTorch/pytorch` | `torch` | `import torch` |
| `scikit-learn/scikit-learn` | `scikit-learn` | `import sklearn` |
| `psf/requests` | `requests` | `import requests` |
| **`PeterPonyu/iAODE`** | **`iaode`** | **`import iaode`** |

## Usage

### Installing
```bash
# Clone repository (mixed case)
git clone https://github.com/PeterPonyu/iAODE.git
cd iAODE

# Install package (lowercase)
pip install -e .

# Or from PyPI
pip install iaode
```

### Using in Python
```python
# Import is always lowercase
import iaode

# Create model
model = iaode.agent(adata, latent_dim=10)
```

## Configuration Files

### pyproject.toml
```toml
[project]
name = "iaode"  # PyPI package name (lowercase)

[project.urls]
Repository = "https://github.com/PeterPonyu/iAODE"  # GitHub repo (mixed case)
```

### Directory Structure
```
iAODE/                  # Repository root (mixed case directory)
├── iaode/             # Package directory (lowercase)
│   ├── __init__.py
│   └── ...
├── setup.py
└── pyproject.toml
```

## No Conflicts

This setup does NOT cause any issues:
- ✅ Git clone works with `iAODE`
- ✅ pip install works with `iaode`
- ✅ Python import works with `iaode`
- ✅ PyPI recognizes `iaode` as the package name
- ✅ GitHub shows `iAODE` as the repository name

## Conclusion

**This is the correct and recommended setup.** The mixed-case repository name (`iAODE`) and lowercase package name (`iaode`) coexist perfectly and follow Python packaging best practices.
