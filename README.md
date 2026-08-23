# gemma4-e2b-text

A one-command, local, text-only Python interface to Google’s quantized Gemma 4
E2B model using LiteRT-LM.

## Install

After the first PyPI release:

```powershell
py -m pip install gemma4-e2b-text
```

No Hugging Face command or separate model package is required. On first use,
the library downloads two assets from this repository's GitHub Release,
assembles the official model directly in the local cache, verifies its SHA-256,
and then runs locally. Later uses work offline.

To download and verify the model before running any prompt:

```powershell
gemma4-e2b --download-only
```

Interrupted downloads resume automatically.

## Python API

```python
from gemma4_e2b_text import Gemma4

with Gemma4(backend="cpu") as model:
    response = model.generate(
        "Florida is a state in which country? Answer in one word."
    )
    print(response)
```

For a single call:

```python
from gemma4_e2b_text import generate

print(generate("Write one sentence about Florida."))
```

## Command line

```powershell
gemma4-e2b "What is the capital of France?"
gemma4-e2b --backend gpu
gemma4-e2b --download-only
```

With no prompt, the command starts a stateful interactive chat. CPU is the
default. GPU automatically enables LiteRT-LM multi-token prediction; NPU is
available where the installed runtime and hardware support it.

## Model download and cache

The official CPU-compatible model is **2,588,147,712 bytes (2.41 GiB)**. GitHub
Release assets must each be under 2 GiB, so the release contains two equal
1,294,073,856-byte parts. The library appends both parts into one temporary
model and only moves it into place after this official SHA-256 passes:

```text
181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c
```

Default cache locations:

- Windows: `%LOCALAPPDATA%\gemma4-e2b-text\models`
- macOS: `~/Library/Caches/gemma4-e2b-text/models`
- Linux: `${XDG_CACHE_HOME:-~/.cache}/gemma4-e2b-text/models`

Set `GEMMA4_E2B_CACHE_DIR` to select another cache directory. Set
`GEMMA4_E2B_MODEL` to use an existing `.litertlm` file and skip downloading.

The 0.8–0.84 GB number published for text-only operation is a working-memory
footprint made possible by memory mapping; it is not the download size.

## Platform support

The LiteRT-LM 0.16.1 Python wheels currently cover Windows x86-64, Linux
x86-64/aarch64, macOS arm64, and Android x86-64/arm64 with Python 3.10+.
Hardware-specific availability may change upstream.

## Source checkout and tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m pip wheel . --no-deps --no-build-isolation -w dist
```

Maintainers publish the model assets with the manual `Publish model assets`
GitHub Actions workflow. It downloads the pinned official artifact, verifies
it, creates the two release parts plus a manifest, and publishes tag
`model-v0.1.0`. See `RELEASING.md`.

## License and safety

The project and upstream model are Apache-2.0 licensed. See `MODEL_NOTICE.md`
for provenance and the pinned hash. This is an independent community package,
not a Google product. Generated text can be inaccurate; validate important
outputs and follow the upstream Gemma responsible-use guidance.

