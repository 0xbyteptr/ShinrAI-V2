#!/usr/bin/env python3
"""Workflow helper for ShinrAI.

Provides one consistent entrypoint for daily tasks:
- environment diagnostics
- training
- corpus stats
- chat

Examples
--------
python scripts/workflow.py doctor
python scripts/workflow.py train --url "https://example.com" --crawl 200
python scripts/workflow.py full --file data/health.json
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHINRAI_PY = ROOT / "shinrai.py"
INFO_PY = ROOT / "info.py"


def run_step(cmd: list[str], title: str) -> int:
    print(f"\n{'=' * 70}\n▶ {title}\n{'=' * 70}")
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(ROOT))
    return result.returncode


def doctor() -> int:
    print(f"Project root: {ROOT}")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform: {platform.platform()}")

    try:
        import torch  # type: ignore

        print(f"Torch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except Exception as err:
        print(f"Torch check failed: {err}")

    hf_cache = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    print(f"HF cache: {hf_cache}")
    return 0


def train(args: argparse.Namespace) -> int:
    cmd = [sys.executable, str(SHINRAI_PY), "train"]

    if args.url:
        cmd.extend(["--url", args.url])
        cmd.extend(["--crawl", str(args.crawl)])
    elif args.file:
        cmd.extend(["--file", args.file])
    elif args.dir:
        cmd.extend(["--dir", args.dir])
    else:
        print("Error: provide one of --url / --file / --dir")
        return 2

    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.lazy:
        cmd.append("--lazy")
    if args.no_topics:
        cmd.append("--no-topics")

    return run_step(cmd, "Train model")


def info() -> int:
    return run_step([sys.executable, str(INFO_PY)], "Corpus/tokenizer stats")


def chat() -> int:
    return run_step([sys.executable, str(SHINRAI_PY), "chat"], "Interactive chat")


def full(args: argparse.Namespace) -> int:
    code = doctor()
    if code != 0:
        return code

    code = train(args)
    if code != 0:
        return code

    code = info()
    if code != 0:
        return code

    if args.chat_after:
        return chat()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ShinrAI workflow helper")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="Print environment and GPU diagnostics")
    sub.add_parser("info", help="Run info.py")
    sub.add_parser("chat", help="Run interactive chat")

    def add_train_like(name: str, help_text: str):
        p = sub.add_parser(name, help=help_text)
        src = p.add_mutually_exclusive_group(required=True)
        src.add_argument("--url", help="Start URL for web scraping")
        src.add_argument("--file", help="Single file path")
        src.add_argument("--dir", help="Directory path")
        p.add_argument("--crawl", type=int, default=100, help="Max pages for URL mode")
        p.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
        p.add_argument("--lazy", action="store_true", help="Enable lazy model initialization")
        p.add_argument("--no-topics", action="store_true", help="Skip topic model training")
        return p

    add_train_like("train", "Train model from URL/file/dir")
    p_full = add_train_like("full", "Run doctor -> train -> info")
    p_full.add_argument("--chat-after", action="store_true", help="Open chat after full workflow")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "doctor":
        return doctor()
    if args.command == "train":
        return train(args)
    if args.command == "info":
        return info()
    if args.command == "chat":
        return chat()
    if args.command == "full":
        return full(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
