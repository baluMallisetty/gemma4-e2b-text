# Gemma 4 E2B model assets

This release contains the official Apache-2.0-licensed
`gemma-4-E2B-it.litertlm` model split into two assets so each remains below
GitHub Releases' 2 GiB per-file limit.

The `gemma4-e2b-text` Python package downloads both parts on first use, joins
them in its local cache, and accepts the result only if it matches:

- Size: `2,588,147,712` bytes
- SHA-256: `181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c`

The original artifact and model card are available from
`litert-community/gemma-4-E2B-it-litert-lm` on Hugging Face. The split does not
modify the model bytes.

