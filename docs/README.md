# pytector Documentation

This directory contains the documentation for the pytector package, built using Sphinx.

## Building Documentation Locally

1. Install the documentation requirements:
   ```bash
   pip install -r docs/requirements.txt
   ```

2. Build the HTML documentation:
   ```bash
   cd docs
   make html
   ```

3. View the documentation:
   ```bash
   open _build/html/index.html
   ```

## Documentation Structure

- `index.rst` - Main documentation homepage
- `installation.rst` - Installation guide
- `quickstart.rst` - Quick start guide
- `api.rst` - API reference
- `examples.rst` - Usage examples
- `contributing.rst` - Contributing guidelines
- `conf.py` - Sphinx configuration
- `requirements.txt` - Documentation dependencies

## Read the Docs Integration

The documentation is configured to work with Read the Docs. The `.readthedocs.yml` file in the root directory contains the configuration for automatic builds.

## Adding New Documentation

1. Create a new `.rst` file in the `docs/` directory
2. Add it to the table of contents in `index.rst`
3. Build and test locally
4. Commit and push to trigger a Read the Docs build

## Documentation Style

- Use reStructuredText (`.rst`) format for most content
- Markdown (`.md`) files are also supported
- Follow the existing style and structure
- Include code examples where appropriate
- Keep documentation up to date with code changes 