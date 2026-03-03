"""LLM training loop — causal language modelling from scratch.

The trainer:
  - tokenises documents with the BPE tokenizer into fixed-length windows
  - trains the GPT model with AdamW + cosine learning-rate schedule
  - uses automatic mixed precision (fp16 on CUDA)
  - clips gradients, logs perplexity, and saves checkpoints

Typical usage
-------------
    from shinrai.llm_tokenizer import BPETokenizer
    from shinrai.llm_model import GPT
    from shinrai.llm_trainer import LLMTrainer, TrainerConfig

    tokenizer = BPETokenizer(vocab_size=16000)
    tokenizer.train(texts)
    model = GPT.small(vocab_size=len(tokenizer))
    cfg = TrainerConfig(epochs=3, batch_size=16, seq_len=512)
    trainer = LLMTrainer(model, tokenizer, cfg, device="cuda")
    trainer.train(texts)
    trainer.save("shinrai_model/llm")
"""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .llm_model import GPT, GPTConfig
from .llm_tokenizer import BPETokenizer


# ── config ────────────────────────────────────────────────────────────────────

@dataclass
class TrainerConfig:
    epochs: int = 3
    batch_size: int = 8
    seq_len: int = 512
    lr: float = 3e-4
    lr_min: float = 3e-5           # cosine decay floor
    warmup_steps: int = 200
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    eval_interval: int = 200       # steps between loss reports
    save_interval: int = 500       # steps between checkpoint saves
    fp16: bool = True              # mixed precision on CUDA


# ── dataset ───────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    """Flattens tokenized documents into overlapping fixed-length windows."""

    def __init__(
        self,
        texts: List[str],
        tokenizer: BPETokenizer,
        seq_len: int = 512,
        stride: Optional[int] = None,
    ):
        self.seq_len = seq_len
        stride = stride or seq_len // 2

        # Build a single long token stream
        all_ids: List[int] = []
        for text in texts:
            ids = tokenizer.encode(text, add_bos=True, add_eos=True)
            all_ids.extend(ids)

        # Cut into windows of (seq_len + 1) for input/target shift
        self.windows: List[List[int]] = []
        win_len = seq_len + 1
        for start in range(0, len(all_ids) - win_len + 1, stride):
            self.windows.append(all_ids[start: start + win_len])

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        x = torch.tensor(window[:-1], dtype=torch.long)  # (seq_len,)
        y = torch.tensor(window[1:],  dtype=torch.long)  # (seq_len,)
        return x, y


# ── trainer ───────────────────────────────────────────────────────────────────

class LLMTrainer:
    """Manages tokenizer training, model training, and checkpointing."""

    def __init__(
        self,
        model: GPT,
        tokenizer: BPETokenizer,
        cfg: TrainerConfig,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Mixed precision scaler (CUDA only)
        self._use_amp = cfg.fp16 and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self._use_amp)

        # Optimizer — exclude weight-tying params from decay
        decay_params = []
        nodecay_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params,   "weight_decay": cfg.weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=cfg.lr,
            betas=(0.9, 0.95),
        )

        self.global_step = 0

    # ── LR schedule ──────────────────────────────────────────────────────────

    def _get_lr(self, total_steps: int) -> float:
        """Cosine decay with linear warmup."""
        step = self.global_step
        if step < self.cfg.warmup_steps:
            return self.cfg.lr * step / max(1, self.cfg.warmup_steps)
        progress = (step - self.cfg.warmup_steps) / max(1, total_steps - self.cfg.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.cfg.lr_min + (self.cfg.lr - self.cfg.lr_min) * cosine

    def _set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    # ── training ─────────────────────────────────────────────────────────────

    def train(self, texts: List[str], save_dir: Optional[str] = None):
        """Full training run.  Tokenizes ``texts``, builds dataset, trains."""
        cfg = self.cfg

        print(f"[LLM] Building dataset from {len(texts):,} documents …")
        dataset = TextDataset(texts, self.tokenizer, seq_len=cfg.seq_len)

        if len(dataset) == 0:
            print("[LLM] Dataset is empty — nothing to train on.")
            return

        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,        # keep it simple / compatible
            pin_memory=(self.device.type == "cuda"),
        )

        total_steps = len(loader) * cfg.epochs
        print(
            f"[LLM] {len(dataset):,} windows | "
            f"{len(loader):,} batches/epoch | "
            f"{total_steps:,} total steps"
        )
        print(
            f"[LLM] Model: {self.model.num_params() / 1e6:.1f}M params | "
            f"device: {self.device} | amp: {self._use_amp}"
        )

        self.model.train()
        t0 = time.time()
        running_loss = 0.0
        best_loss = float("inf")

        for epoch in range(1, cfg.epochs + 1):
            for batch_idx, (x, y) in enumerate(loader):
                x = x.to(self.device)
                y = y.to(self.device)

                # LR schedule
                lr = self._get_lr(total_steps)
                self._set_lr(lr)

                # Forward
                dtype = torch.float16 if self._use_amp else torch.float32
                with torch.autocast(device_type=self.device.type, dtype=dtype,
                                    enabled=self._use_amp):
                    _, loss, _ = self.model(x, targets=y)

                # Backward
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                self.global_step += 1
                running_loss += loss.item()

                # Logging
                if self.global_step % cfg.eval_interval == 0:
                    avg = running_loss / cfg.eval_interval
                    ppl = math.exp(min(avg, 20))
                    elapsed = time.time() - t0
                    steps_per_sec = cfg.eval_interval / elapsed
                    print(
                        f"  epoch {epoch}/{cfg.epochs}  "
                        f"step {self.global_step:6d}/{total_steps}  "
                        f"loss {avg:.4f}  ppl {ppl:.1f}  "
                        f"lr {lr:.2e}  "
                        f"{steps_per_sec:.1f} steps/s"
                    )
                    if avg < best_loss:
                        best_loss = avg
                    running_loss = 0.0
                    t0 = time.time()

                # Checkpoint
                if save_dir and self.global_step % cfg.save_interval == 0:
                    self.save(save_dir, tag=f"step{self.global_step}")

        print(f"[LLM] Training complete. Best loss: {best_loss:.4f}  "
              f"(ppl {math.exp(min(best_loss, 20)):.1f})")

        if save_dir:
            self.save(save_dir)

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, directory: str, tag: str = "final"):
        """Save model weights + config + tokenizer to *directory*."""
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"llm_{tag}.pt")
        torch.save({
            "model_state": self.model.state_dict(),
            "model_cfg":   self.model.cfg,
            "global_step": self.global_step,
        }, path)
        # Always save tokenizer alongside weights
        tok_path = os.path.join(directory, "llm_tokenizer.json")
        self.tokenizer.save(tok_path)
        print(f"[LLM] Saved checkpoint → {path}")

    @staticmethod
    def load_model_from(directory: str, tag: str = "final", device: str = "cuda"):
        """Load and return a GPT model from a checkpoint directory."""
        path = os.path.join(directory, f"llm_{tag}.pt")
        ck = torch.load(path, map_location=device)
        cfg: GPTConfig = ck["model_cfg"]
        model = GPT(cfg)
        model.load_state_dict(ck["model_state"])
        model.to(device)
        return model, ck.get("global_step", 0)
