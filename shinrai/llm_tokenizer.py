"""Byte-Pair Encoding (BPE) tokenizer — built entirely from scratch.

Uses **byte-level** BPE (à la GPT-2): words are first converted to sequences
of raw bytes (0-255), so the base alphabet is always exactly 256 symbols
regardless of how many unique Unicode characters appear in the corpus.  This
means vocab_size only needs to be > 256 + 4 special tokens = 260, and works
correctly on any language or character set.

No external tokenizer libraries are required.  The tokenizer:
  - trains a merge table on raw text corpus
  - handles special tokens (<pad>, <bos>, <eos>, <unk>)
  - encodes/decodes with the trained vocabulary
  - saves/loads from a single JSON file

Typical usage
-------------
    tok = BPETokenizer(vocab_size=16000)
    tok.train(list_of_strings)
    ids = tok.encode("hello world")
    text = tok.decode(ids)
    tok.save("shinrai_model/tokenizer.json")
    tok2 = BPETokenizer.load("shinrai_model/tokenizer.json")
"""

from __future__ import annotations

import json
import re
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# ── special token constants ──────────────────────────────────────────────────
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

# ── byte-level helpers ───────────────────────────────────────────────────────
# Represent each raw byte as a 2-character hex string like "00", "ff" so they
# are unambiguous single symbols that will never clash with real text tokens.

def _byte_symbol(b: int) -> str:
    """Map a byte value 0-255 to a unique 2-char hex symbol."""
    return f"\\x{b:02x}"


def _word_to_byte_symbols(word: str) -> Tuple[str, ...]:
    """Encode a word string as a tuple of byte symbols."""
    return tuple(_byte_symbol(b) for b in word.encode("utf-8", errors="replace"))


def _byte_symbols_to_str(symbols: List[str]) -> str:
    """Decode a list of (possibly merged) byte symbols back to a Python string.

    Merged tokens are strings like "\\x68\\x65\\x6c" — we split them back into
    individual byte symbols and decode.
    """
    # split each symbol back into its constituent 4-char hex units
    raw: List[int] = []
    for sym in symbols:
        # each symbol is one or more "\\xNN" pieces
        i = 0
        while i < len(sym):
            if sym[i:i+2] == "\\x" and i + 4 <= len(sym):
                raw.append(int(sym[i+2:i+4], 16))
                i += 4
            else:
                # shouldn't happen with well-formed tokens
                i += 1
    return bytes(raw).decode("utf-8", errors="replace")


def _pretokenize(text: str) -> List[str]:
    """Split raw text into words using a simple regex (GPT-2 style split)."""
    pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+|[^\s\w\d]+"""
    return re.findall(pattern, text)


class BPETokenizer:
    """Byte-level Byte-Pair Encoding tokenizer trained on a text corpus.

    The base alphabet is always exactly 256 byte symbols so vocab_size only
    needs to be > 260 (256 bytes + 4 special tokens).
    """

    # Minimum vocab: 4 special + 256 bytes
    MIN_VOCAB_SIZE = 260

    def __init__(self, vocab_size: int = 16000):
        if vocab_size < self.MIN_VOCAB_SIZE:
            raise ValueError(
                f"vocab_size must be at least {self.MIN_VOCAB_SIZE} "
                f"(4 special tokens + 256 byte symbols); got {vocab_size}"
            )
        self.vocab_size = vocab_size
        # maps token string → int id
        self.token_to_id: Dict[str, int] = {}
        # maps int id → token string
        self.id_to_token: Dict[int, str] = {}
        # ordered merge rules: list of (left, right) tuples
        self.merges: List[Tuple[str, str]] = []
        self._trained = False

    # ── training ─────────────────────────────────────────────────────────────

    # Maximum documents to sample for tokenizer training.
    # The BPE vocabulary is language-statistical, not document-specific, so
    # a random 5 000-doc sample produces essentially the same merge table as
    # the full corpus while being much faster.
    TRAIN_SAMPLE_DOCS = 5000

    def train(self, texts: List[str], verbose: bool = True):
        """Learn BPE merges from a list of text strings.

        Uses an **incremental pair-count** algorithm:
          1. Build pair_counts and a pair→word-indices inverted index once.
          2. Each merge step only visits the O(affected_words) words that
             actually contain the chosen pair — not the entire vocabulary.
        This reduces complexity from O(steps × corpus) to roughly
        O(steps × mean_affected_words), making 15 k steps tractable.
        """
        import random as _random

        # 1. build fixed byte-level base vocabulary
        base_vocab: Dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for b in range(256):
            sym = _byte_symbol(b)
            if sym not in base_vocab:
                base_vocab[sym] = len(base_vocab)

        # 2. optionally sample a subset of documents
        sample = texts
        if len(texts) > self.TRAIN_SAMPLE_DOCS:
            sample = _random.sample(texts, self.TRAIN_SAMPLE_DOCS)
            if verbose:
                print(
                    f"[BPE] Sampling {self.TRAIN_SAMPLE_DOCS:,} / {len(texts):,} "
                    "documents for tokenizer training …",
                    flush=True,
                )

        if verbose:
            print(f"[BPE] Pre-tokenizing {len(sample):,} documents …", flush=True)

        # word frequencies: tuple-of-byte-symbols → count
        raw_freq: Dict[Tuple[str, ...], int] = defaultdict(int)
        for text in sample:
            for word in _pretokenize(text):
                if word:
                    raw_freq[_word_to_byte_symbols(word)] += 1

        if verbose:
            total_words = sum(raw_freq.values())
            print(
                f"[BPE] {total_words:,} word occurrences, "
                f"{len(raw_freq):,} unique words, "
                f"base vocab = {len(base_vocab)} byte symbols",
                flush=True,
            )

        # Convert to indexed arrays for O(1) lookup during incremental updates
        word_seqs: List[List[str]] = [list(s) for s in raw_freq.keys()]
        word_freqs: List[int] = list(raw_freq.values())
        n_words = len(word_seqs)

        # Build initial pair counts and inverted index: pair → set of word indices
        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        pair_to_idx: Dict[Tuple[str, str], set] = defaultdict(set)

        for idx in range(n_words):
            seq = word_seqs[idx]
            freq = word_freqs[idx]
            for i in range(len(seq) - 1):
                p = (seq[i], seq[i + 1])
                pair_counts[p] += freq
                pair_to_idx[p].add(idx)

        # 3. iterative merge loop
        n_merges = self.vocab_size - len(base_vocab)
        vocab = dict(base_vocab)
        merges: List[Tuple[str, str]] = []

        for step in range(n_merges):
            if not pair_counts:
                break

            best = max(pair_counts, key=pair_counts.__getitem__)
            if pair_counts[best] < 2:
                break

            left, right = best
            merged = left + right
            if merged not in vocab:
                vocab[merged] = len(vocab)
            merges.append(best)

            # Only update words that contain the best pair
            affected = list(pair_to_idx.get(best, set()))
            for idx in affected:
                seq = word_seqs[idx]
                freq = word_freqs[idx]

                new_seq: List[str] = []
                i = 0
                while i < len(seq):
                    if i < len(seq) - 1 and seq[i] == left and seq[i + 1] == right:
                        # ── remove neighbor pair contributions ──────────────
                        if new_seq:
                            lp = (new_seq[-1], left)
                            pair_counts[lp] -= freq
                            if pair_counts[lp] <= 0:
                                pair_counts.pop(lp, None)
                            pair_to_idx[lp].discard(idx)

                        if i + 2 < len(seq):
                            rp = (right, seq[i + 2])
                            pair_counts[rp] -= freq
                            if pair_counts[rp] <= 0:
                                pair_counts.pop(rp, None)
                            pair_to_idx[rp].discard(idx)

                        new_seq.append(merged)

                        # ── add new neighbor pair contributions ─────────────
                        if len(new_seq) >= 2:
                            nlp = (new_seq[-2], merged)
                            pair_counts[nlp] += freq
                            pair_to_idx[nlp].add(idx)

                        if i + 2 < len(seq):
                            nrp = (merged, seq[i + 2])
                            pair_counts[nrp] += freq
                            pair_to_idx[nrp].add(idx)

                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1

                word_seqs[idx] = new_seq

            # Remove the consumed pair entry entirely
            pair_counts.pop(best, None)
            pair_to_idx.pop(best, None)

            if verbose and (step + 1) % 500 == 0:
                print(
                    f"[BPE] Step {step + 1:5d}/{n_merges}  "
                    f"vocab={len(vocab):,}",
                    flush=True,
                )

        self.token_to_id = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}
        self.merges = merges
        self._trained = True

        self._merge_rank: Dict[Tuple[str, str], int] = {
            pair: rank for rank, pair in enumerate(merges)
        }
        if verbose:
            print(f"[BPE] Training complete. Vocab size: {len(self.token_to_id):,}")

    # ── encoding ─────────────────────────────────────────────────────────────

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """Encode *text* to a list of integer token ids."""
        ids: List[int] = []
        if add_bos:
            ids.append(BOS_ID)
        for word in _pretokenize(text):
            if not word:
                continue
            ids.extend(self._encode_word(word))
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def _encode_word(self, word: str) -> List[int]:
        """BPE-encode a single pre-tokenized word."""
        seq = list(_word_to_byte_symbols(word))
        # apply merges greedily in rank order
        while len(seq) >= 2:
            best_rank = len(self.merges) + 1
            best_i = -1
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                rank = self._merge_rank.get(pair, len(self.merges) + 1)
                if rank < best_rank:
                    best_rank = rank
                    best_i = i
            if best_i == -1:
                break
            left = seq[best_i]
            right = seq[best_i + 1]
            seq = seq[:best_i] + [left + right] + seq[best_i + 2:]

        return [self.token_to_id.get(tok, UNK_ID) for tok in seq]

    def encode_batch(self, texts: List[str], **kwargs) -> List[List[int]]:
        return [self.encode(t, **kwargs) for t in texts]

    # ── decoding ─────────────────────────────────────────────────────────────

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode a list of token ids back to a string."""
        skip_set = set(SPECIAL_TOKENS) if skip_special else set()
        symbols: List[str] = []
        for i in ids:
            tok = self.id_to_token.get(i, UNK_TOKEN)
            if tok in skip_set:
                continue
            symbols.append(tok)
        return _byte_symbols_to_str(symbols)

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: str):
        """Serialize tokenizer to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
            "merges": self.merges,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load a serialized tokenizer from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(vocab_size=data["vocab_size"])
        tok.token_to_id = data["token_to_id"]
        tok.id_to_token = {int(v): k for k, v in tok.token_to_id.items()}
        tok.merges = [tuple(p) for p in data["merges"]]
        tok._merge_rank = {tuple(p): rank for rank, p in enumerate(tok.merges)}
        tok._trained = True
        return tok

    # ── helpers ──────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID


# ── helpers ──────────────────────────────────────────────────────────────────

def _apply_merge(seq: Tuple[str, ...], left: str, right: str) -> Tuple[str, ...]:
    """Return a new sequence with every occurrence of (left, right) merged."""
    result: List[str] = []
    i = 0
    while i < len(seq):
        if i < len(seq) - 1 and seq[i] == left and seq[i + 1] == right:
            result.append(left + right)
            i += 2
        else:
            result.append(seq[i])
            i += 1
    return tuple(result)

