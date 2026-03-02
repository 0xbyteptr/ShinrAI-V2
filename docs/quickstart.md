# Quick Start

## 1) Requirements

- Python 3.10+
- Linux/macOS/Windows (WSL works too)
- Internet access (for first model downloads)

## 2) Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3) Train the model

### A. Train from URL

```bash
python shinrai.py train --url "https://example.com" --crawl 100
```

### B. Train from file

```bash
python shinrai.py train --file data/health.json
```

### C. Train from directory

```bash
python shinrai.py train --dir data/
```

### Useful options

- `--crawl N` — maximum number of pages to crawl
- `--batch-size N` — embedding batch size
- `--lazy` — delayed model initialization
- `--no-topics` — skip topic model training

## 4) Interactive chat

```bash
python shinrai.py chat
```

Chat commands:

- `/help`
- `/memory`
- `/graph`
- `/topics`
- `/save`
- `/clear`
- `/exit`

## 5) Corpus/tokenizer stats

```bash
python info.py
```

Example output fields:

- `Documents in corpus`
- `Tokenizer class`
- `Vocabulary size`
- `Total tokens across documents`
- `Unique token ids in corpus`

## 6) Discord (optional)

Fill in `discord_config.json`, then run:

```bash
python dc.py
```

## 7) Improved workflow helper

Instead of running multiple commands manually, use:

```bash
python scripts/workflow.py doctor
python scripts/workflow.py train --url "https://example.com" --crawl 100
python scripts/workflow.py info
python scripts/workflow.py chat
```

Run the complete flow in one command:

```bash
python scripts/workflow.py full --url "https://example.com" --crawl 100
```

Optional: open chat automatically after the full flow:

```bash
python scripts/workflow.py full --url "https://example.com" --crawl 100 --chat-after
```
