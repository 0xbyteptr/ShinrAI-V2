#!/usr/bin/env python3
"""Utility script that prints model/tokenizer statistics.

This is a lightweight helper the user requested; run from the
project root like:

    python info.py

The script will load the tokenizer (without downloading the full
transformer weights) and report:

* vocabulary size
* number of tokens currently stored in the document corpus
* number of unique token ids in that corpus (if any documents have been
  added via training)
* number of documents loaded into the Shinrai instance

Additional details can be added later, but these metrics provide a
quick sanity check when inspecting or debugging the model directory.
"""

import sys
from pathlib import Path

from shinrai.core import Shinrai


def human_format(num: int) -> str:
    """Return a human-readable string for a large integer."""
    for unit in ['', 'K', 'M', 'B']:
        if abs(num) < 1000.0:
            return f"{num:.0f}{unit}"
        num /= 1000.0
    return f"{num:.1f}T"


def main():
    # create Shinrai lazily so we don't pay the cost of loading the full
    # transformer model unless absolutely necessary.
    shinrai = Shinrai(lazy=True)
    shinrai._ensure_models()

    tok = shinrai.tokenizer

    # report corpus even if tokenizer is missing
    num_docs = len(shinrai.documents)
    print(f"Documents in corpus: {num_docs}")

    if tok is None:
        print("Tokenizer is not available; vocabulary/token counts unavailable.")
    else:
        # vocabulary size is exposed a couple of different ways depending on
        # tokenizer implementation; be defensive.
        vocab_size = getattr(tok, 'vocab_size', None) or len(getattr(tok, 'get_vocab', lambda: tok)())
        print(f"Tokenizer class: {tok.__class__.__name__}")
        print(f"Vocabulary size: {vocab_size}")

        if num_docs:
            enc = tok(shinrai.documents, truncation=False, padding=False)
            all_ids = enc.get('input_ids', [])
            total_tokens = sum(len(ids) for ids in all_ids)
            unique_tokens = len(set(tok_id for seq in all_ids for tok_id in seq))
            print(f"Total tokens across documents: {total_tokens} ({human_format(total_tokens)})")
            print(f"Unique token ids in corpus: {unique_tokens}")
        else:
            print("No documents currently loaded; token counts unavailable.")


if __name__ == '__main__':
    main()
