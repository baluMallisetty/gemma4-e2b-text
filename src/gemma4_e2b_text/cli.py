"""Command-line interface for gemma4-e2b-text."""

from __future__ import annotations

import argparse
import sys

from .model import Gemma4


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gemma4-e2b",
        description="Run bundled Gemma 4 E2B locally with a text-only interface.",
    )
    parser.add_argument("prompt", nargs="*", help="Prompt; omit for interactive chat")
    parser.add_argument("--system", help="Optional system instruction")
    parser.add_argument("--backend", choices=("cpu", "gpu", "npu"), default="cpu")
    parser.add_argument("--model", help="Override the bundled .litertlm path")
    parser.add_argument("--cache-dir", help="LiteRT-LM compilation cache directory")
    parser.add_argument("--no-stream", action="store_true", help="Wait for the full response")
    return parser


def _print_response(model: Gemma4, prompt: str, system: str | None, stream: bool) -> None:
    if stream:
        for fragment in model.stream(prompt, system=system):
            print(fragment, end="", flush=True)
        print()
    else:
        print(model.generate(prompt, system=system))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        with Gemma4(
            model_path=args.model,
            backend=args.backend,
            cache_dir=args.cache_dir,
        ) as model:
            if args.prompt:
                _print_response(model, " ".join(args.prompt), args.system, not args.no_stream)
                return 0

            print("Gemma 4 E2B local chat. Type /exit to quit.")
            with model.conversation(system=args.system) as conversation:
                while True:
                    try:
                        prompt = input("\nYou: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print()
                        break
                    if prompt.lower() in {"/exit", "/quit"}:
                        break
                    if not prompt:
                        continue
                    print("AI: ", end="", flush=True)
                    for fragment in conversation.stream(prompt):
                        print(fragment, end="", flush=True)
                    print()
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"gemma4-e2b: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

