# Releasing

The source tree is release-ready, but publication requires control of the PyPI
projects and an approved file-size increase for the weights project.

## 1. Replace repository placeholders

Confirm the PyPI names are still available immediately before the first upload:

```bash
python -m pip index versions gemma4-e2b-text
python -m pip index versions gemma4-e2b-text-weights
```

## 2. Create the weights project under PyPI’s default limit

Build and upload only the developmental placeholder. It is deliberately marked
as a pre-release and its `model_path()` raises a clear error.

```bash
python scripts/build_weights_wheel.py --placeholder --version 0.1.0.dev0
python scripts/verify_wheel.py dist/gemma4_e2b_text_weights-0.1.0.dev0-py3-none-any.whl
python -m twine check dist/gemma4_e2b_text_weights-0.1.0.dev0-py3-none-any.whl
python -m twine upload dist/gemma4_e2b_text_weights-0.1.0.dev0-py3-none-any.whl
```

Do not paste a PyPI token into source, chat, or a command saved in shell history.
Prefer a PyPI Trusted Publisher tied to the GitHub Actions environment `pypi`.

## 3. Request the large-file exception

Open a PyPI support issue using `PYPI_SIZE_LIMIT_REQUEST.md`. Ask for a
**2,700 MB individual-file limit** on `gemma4-e2b-text-weights`. Wait for
approval before attempting the stable upload.

## 4. Build the pinned stable weights wheel

The downloader and builder both verify the official size and SHA-256. The wheel
uses ZIP64 with stored (uncompressed) model data, avoiding a second temporary
copy during compression.

```bash
python scripts/download_model.py --destination model
python scripts/build_weights_wheel.py --model model/gemma-4-E2B-it.litertlm --version 0.1.0
python scripts/verify_wheel.py dist/gemma4_e2b_text_weights-0.1.0-py3-none-any.whl
```

Allow at least 8 GB of free disk space for the source model, wheel, runtime
artifacts, and safety margin.

## 5. Build and publish in dependency order

```bash
python -m build --wheel --sdist
python -m twine check dist/*
python -m twine upload dist/gemma4_e2b_text_weights-0.1.0-py3-none-any.whl
python -m twine upload dist/gemma4_e2b_text-0.1.0-py3-none-any.whl dist/gemma4_e2b_text-0.1.0.tar.gz
```

The included GitHub workflow performs the same sequence with PyPI Trusted
Publishing. Configure the PyPI projects to trust:

- Owner/repository: `baluMallisetty/gemma4-e2b-text`
- Workflow: `publish.yml`
- Environment: `pypi`

## 6. Smoke test from a clean environment

```bash
python -m venv clean-test
clean-test/Scripts/python -m pip install --no-cache-dir gemma4-e2b-text==0.1.0
clean-test/Scripts/gemma4-e2b "Answer with one word: What country is Florida in?"
```

On Linux/macOS, use `clean-test/bin/python` and `clean-test/bin/gemma4-e2b`.
