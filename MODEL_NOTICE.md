# Model notice

This project redistributes the official `gemma-4-E2B-it.litertlm` artifact from
[`litert-community/gemma-4-E2B-it-litert-lm`](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm)
as two byte-for-byte GitHub Release parts.

- Upstream license: Apache License 2.0
- Upstream model: Google Gemma 4 E2B instruction-tuned
- Runtime format: LiteRT-LM
- Complete file size: `2,588,147,712` bytes
- Complete SHA-256: `181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c`
- Original artifact: `gemma-4-E2B-it.litertlm`
- Package release: `model-v0.1.0`

Splitting is a transport operation only; joining the two release assets
reproduces the exact upstream bytes. The Python package validates the complete
official hash before activating the cached model.

The wrapper is an independent community project. It is not affiliated with or
endorsed by Google. “Gemma” is used only to identify the upstream model.

The package exposes a text-only API. The official CPU-compatible LiteRT-LM
artifact may contain optional multimodal modules. LiteRT-LM loads those modules
on demand; this wrapper does not initialize vision or audio backends. The
often-quoted 0.8–0.84 GB figure is a text-generation working-memory footprint,
not the download size.

