# gemma4-e2b-text

A one-command, local, text-only Python interface to Google’s quantized Gemma 4
E2B model using LiteRT-LM.

Once the two PyPI projects are published:

```powershell
py -m pip install gemma4-e2b-text
```

That single command installs the wrapper, the matching LiteRT-LM runtime, and
the verified model-weights wheel. Use `--no-cache-dir` if you do not want pip to
retain another copy of the large weights wheel:

```powershell
py -m pip install --no-cache-dir gemma4-e2b-text
```

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
```

With no prompt, the command starts a stateful interactive chat. CPU is the
default. GPU automatically enables LiteRT-LM multi-token prediction; NPU is
available where the installed runtime and hardware support it.

## Actual size and platform support

The official CPU-compatible model file is **2,588,147,712 bytes (2.59 GB
decimal)**. The 0.8–0.84 GB number published for text-only use describes the
working set made possible by memory mapping, not the file or pip-download size.

The LiteRT-LM 0.16.1 Python wheels currently cover Windows x86-64, Linux
x86-64/aarch64, macOS arm64, and Android x86-64/arm64 with Python 3.10+.
Hardware-specific availability may change upstream.

## Source checkout and tests

The source archive intentionally omits the 2.59 GB model. Run the wrapper tests
without downloading it:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m pip wheel . --no-deps --no-build-isolation -w dist
python scripts/build_weights_wheel.py --placeholder --version 0.1.0.dev0
python scripts/verify_wheel.py dist/gemma4_e2b_text_weights-0.1.0.dev0-py3-none-any.whl
```

See `RELEASING.md` for the guarded stable build and PyPI publication sequence.

## License and safety

The project and upstream model are Apache-2.0 licensed. See `MODEL_NOTICE.md`
for exact provenance and the pinned hash. Generated text can be inaccurate;
validate important outputs and follow the upstream Gemma responsible-use
guidance.
