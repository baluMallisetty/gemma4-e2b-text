# Model notice

The stable `gemma4-e2b-text-weights` wheel is designed to redistribute the
official `gemma-4-E2B-it.litertlm` artifact from
[`litert-community/gemma-4-E2B-it-litert-lm`](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm).

- Upstream license: Apache License 2.0
- Upstream model: Google Gemma 4 E2B instruction-tuned
- Runtime format: LiteRT-LM
- Expected file size: `2,588,147,712` bytes
- Expected SHA-256: `181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c`
- Source URL: `https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/blob/main/gemma-4-E2B-it.litertlm`

The Python wrapper in this repository is an independent community project. It
is not affiliated with or endorsed by Google. “Gemma” is used only to identify
the upstream model.

The package exposes a text-only API. The official CPU-compatible LiteRT-LM
artifact is about 2.59 GB on disk and can contain optional multimodal modules.
LiteRT-LM loads those optional modules on demand; this wrapper does not
initialize vision or audio backends. The often-quoted 0.8–0.84 GB figure is a
text-generation working-memory footprint, not the download size.

