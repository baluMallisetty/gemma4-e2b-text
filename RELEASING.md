# Releasing

This project uses two independent releases:

1. GitHub Release `model-v0.1.0` hosts two model parts.
2. PyPI project `gemma4-e2b-text` hosts the small Python wrapper.

The Python package downloads and verifies the GitHub Release assets on first
use. No large PyPI file exception and no user-facing Hugging Face download are
required.

## 1. Publish the model release

In GitHub, open **Actions → Publish model assets → Run workflow** on `main`.
The workflow:

- downloads the official `gemma-4-E2B-it.litertlm` artifact;
- verifies the pinned size and SHA-256;
- splits it into two 1,294,073,856-byte assets;
- creates `model-manifest.json` with part and complete-model hashes;
- creates or safely updates GitHub Release `model-v0.1.0`.

The job needs approximately 7 GB of temporary runner disk. It publishes only
after all verification and splitting steps pass.

Confirm the release contains exactly:

```text
gemma-4-E2B-it.litertlm.part-001
gemma-4-E2B-it.litertlm.part-002
model-manifest.json
```

## 2. Test the public model path

From a clean source checkout:

```bash
PYTHONPATH=src python -c "from gemma4_e2b_text import download_model; print(download_model())"
```

The resulting model must be `2,588,147,712` bytes with SHA-256:

```text
181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c
```

## 3. Configure PyPI Trusted Publishing

Configure the PyPI project to trust:

- Owner/repository: `baluMallisetty/gemma4-e2b-text`
- Workflow: `publish.yml`
- Environment: `pypi`

Do not paste a PyPI token into source, chat, workflow inputs, or shell history.

## 4. Publish the Python package

In GitHub, open **Actions → Publish to PyPI → Run workflow** on `main`. The
workflow runs unit tests, builds the wheel and source distribution, checks both,
and publishes through OpenID Connect.

## 5. Smoke test

```bash
python -m venv clean-test
clean-test/Scripts/python -m pip install --no-cache-dir gemma4-e2b-text==0.1.0
clean-test/Scripts/gemma4-e2b --download-only
clean-test/Scripts/gemma4-e2b "Answer with one word: What country is Florida in?"
```

On Linux/macOS, use `clean-test/bin/python` and `clean-test/bin/gemma4-e2b`.

## Updating the model

A different upstream model requires a new package version and model release
tag. Update, test, and review these together:

- `scripts/model_info.py`
- `scripts/split_model.py`
- `src/gemma4_e2b_text/download.py`
- `.github/workflows/publish-model.yml`
- `MODEL_NOTICE.md` and `MODEL_RELEASE_NOTES.md`

Never replace release assets without updating the pinned complete-model hash
when the model bytes have actually changed.

