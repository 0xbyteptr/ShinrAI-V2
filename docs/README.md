# ShinrAI Documentation

This directory contains structured documentation for **ShinrAI-V2**.

## Table of Contents

- [Quick Start](quickstart.md) — fast setup and run (train/chat/info)
- [Troubleshooting](troubleshooting.md) — common issues and fixes

## Recommended workflow

Use the workflow helper for a consistent daily routine:

```bash
python scripts/workflow.py doctor
python scripts/workflow.py train --url "https://example.com" --crawl 100
python scripts/workflow.py info
python scripts/workflow.py chat
```

One-command run (diagnostics + train + stats):

```bash
python scripts/workflow.py full --url "https://example.com" --crawl 100
```

## Who is this for?

- users who want to run the bot locally,
- users training the model from URLs and files,
- users troubleshooting model, scraping, or tokenizer issues.

## Main project entrypoints

- `shinrai.py` — main CLI (`train`, `chat`)
- `info.py` — corpus/tokenizer statistics
- `dc.py` — Discord integration

## Quick commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python shinrai.py train --url "https://example.com" --crawl 100
python shinrai.py chat
python info.py
```
