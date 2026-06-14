# Publishing `csp-sdk` to PyPI

The package is configured and builds clean. These are the steps to release.

## One-time setup

1. Create accounts: <https://pypi.org/account/register/> and
   (recommended for a dry run) <https://test.pypi.org/account/register/>.
2. Create an API token: PyPI → Account settings → **API tokens** → "Add API
   token" (scope: entire account for the first upload, then project-scoped).
3. Install tooling (already in the dev venv):

   ```bash
   .venv/bin/pip install build twine
   ```

## Build

```bash
rm -rf dist build
.venv/bin/python -m build        # → dist/csp_sdk-0.1.0.tar.gz + .whl
.venv/bin/twine check dist/*     # must say PASSED
```

## Dry run on TestPyPI (recommended)

```bash
.venv/bin/twine upload --repository testpypi dist/*
# username: __token__
# password: <your TestPyPI token, including the "pypi-" prefix>

# verify in a clean venv:
python -m venv /tmp/t && /tmp/t/bin/pip install \
  -i https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  "csp-sdk[langgraph]"
```

(The `--extra-index-url` lets TestPyPI pull real deps like `anthropic`.)

## Publish to PyPI

```bash
.venv/bin/twine upload dist/*
# username: __token__
# password: <your PyPI token>
```

Then anyone can:

```bash
pip install csp-sdk
pip install "csp-sdk[langgraph]"
```

## Notes

- **Name availability:** confirm `csp-sdk` is free at
  <https://pypi.org/project/csp-sdk/>. If taken, change `name` in
  `pyproject.toml` (the import name stays `csp`).
- **Versioning:** PyPI refuses to overwrite a version. Bump `version` in
  `pyproject.toml` for every release (e.g. `0.1.1`), rebuild, re-upload.
- **What ships:** only the `csp/` package (verified via the wheel contents).
  The demo apps (`helloworld/`, `algoviz/`), `tests/`, and `examples/` are
  excluded by the `[tool.hatch.build.targets.sdist]` include list.
- **Credentials:** store the token in `~/.pypirc` or a `TWINE_PASSWORD` env var;
  never commit it.
