# PyPI file-size limit request

Suggested issue title:

> File size limit increase for gemma4-e2b-text-weights to 2700 MB

Suggested issue body (replace the bracketed values):

> Project URL: https://pypi.org/project/gemma4-e2b-text-weights/
>
> Requested limit: 2700 MB per file on PyPI. [Add TestPyPI too if needed.]
>
> `gemma4-e2b-text-weights` contains the Apache-2.0-licensed, official Gemma 4
> E2B LiteRT-LM artifact used by the companion `gemma4-e2b-text` Python
> package. The pinned upstream file is 2,588,147,712 bytes and has SHA-256
> `181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c`.
> Bundling it lets `pip install gemma4-e2b-text` produce a fully local runtime
> without a separate model download or post-install network execution. The
> wrapper and model are separated so wrapper releases do not duplicate the
> model payload. A developmental placeholder release under the default limit
> has already been uploaded as required.
>
> Upstream model: https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm

PyPI’s current documented defaults are 100 MB per uploaded file and 10 GB per
project. A small release must exist before requesting an individual-file limit
increase.

