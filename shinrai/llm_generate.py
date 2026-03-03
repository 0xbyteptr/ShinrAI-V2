"""Autoregressive text generation for the Shinrai GPT model.

Sampling strategies
-------------------
Temperature : rescales logits before softmax (lower = more focused)
Top-k       : keeps only the k highest-probability tokens
Top-p       : nucleus sampling — keeps the smallest set of tokens whose
              cumulative probability ≥ p

The generator also supports a "RAG prefix" mode where retrieved documents
are prepended to the prompt so the model can condition on relevant knowledge.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import List, Optional

from .llm_model import GPT
from .llm_tokenizer import BPETokenizer


class LLMGenerator:
    """Wraps a trained GPT for interactive next-token generation."""

    def __init__(
        self,
        model: GPT,
        tokenizer: BPETokenizer,
        device: str = "cuda",
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_seq_len = model.cfg.max_seq_len

    # ── public API ───────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        stop_at_eos: bool = True,
        context_docs: Optional[List[str]] = None,
    ) -> str:
        """Generate a continuation for *prompt*.

        Parameters
        ----------
        prompt:
            The user-visible prompt / question.
        max_new_tokens:
            Maximum number of new tokens to generate.
        temperature:
            Sampling temperature (1.0 = unchanged, <1.0 = sharper, >1.0 = more random).
        top_k:
            If > 0, keep only top-k logits before sampling.
        top_p:
            Nucleus probability threshold.  Set to 1.0 to disable.
        repetition_penalty:
            Penalise tokens that already appear in the context (> 1.0 = penalise).
        stop_at_eos:
            Stop generation when the EOS token is emitted.
        context_docs:
            Optional list of retrieved documents to prepend as "context" (RAG).
        """
        # Build the full prompt string
        full_prompt = self._build_prompt(prompt, context_docs)

        input_ids = self.tokenizer.encode(full_prompt, add_bos=True)

        # Truncate from the left if prompt already exceeds model limit
        if len(input_ids) >= self.max_seq_len:
            input_ids = input_ids[-(self.max_seq_len - max_new_tokens - 1):]

        ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        prompt_len = ids.shape[1]
        past_kvs = None

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                # Use only the last token when using KV cache
                if past_kvs is not None:
                    x_in = ids[:, -1:]
                else:
                    x_in = ids

                if x_in.shape[1] > self.max_seq_len:
                    x_in = x_in[:, -self.max_seq_len:]

                logits, _, new_kvs = self.model(x_in, past_kvs=past_kvs)
                past_kvs = new_kvs

                # Logits for next token
                next_logits = logits[:, -1, :]   # (1, vocab_size)

                # Repetition penalty
                if repetition_penalty != 1.0:
                    for token_id in set(ids[0].tolist()):
                        if next_logits[0, token_id] < 0:
                            next_logits[0, token_id] *= repetition_penalty
                        else:
                            next_logits[0, token_id] /= repetition_penalty

                # Temperature
                if temperature != 1.0:
                    next_logits = next_logits / temperature

                # Top-k
                if top_k > 0:
                    next_logits = _top_k_filter(next_logits, top_k)

                # Top-p (nucleus)
                if top_p < 1.0:
                    next_logits = _top_p_filter(next_logits, top_p)

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                if stop_at_eos and next_token.item() == self.tokenizer.eos_id:
                    break

                ids = torch.cat([ids, next_token], dim=1)

        generated_ids = ids[0, prompt_len:].tolist()
        return self.tokenizer.decode(generated_ids)

    # ── prompt construction ───────────────────────────────────────────────────

    def _build_prompt(
        self,
        query: str,
        context_docs: Optional[List[str]] = None,
    ) -> str:
        """Build the full prompt, optionally injecting retrieved documents."""
        if not context_docs:
            return f"Human: {query}\nAssistant:"

        # Take up to 3 context docs, truncated to keep total prompt manageable
        ctx_parts = []
        for doc in context_docs[:3]:
            snippet = doc[:400].strip()
            if snippet:
                ctx_parts.append(snippet)

        context_str = "\n\n".join(ctx_parts)
        return (
            f"Context:\n{context_str}\n\n"
            f"Human: {query}\nAssistant:"
        )


# ── sampling helpers ──────────────────────────────────────────────────────────

def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out all logits except the top-k."""
    if k == 0:
        return logits
    values, _ = torch.topk(logits, min(k, logits.size(-1)))
    threshold = values[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < threshold, float("-inf"))


def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Zero out logits outside the nucleus (top-p) probability mass."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens above the threshold (shift right so we keep the boundary token)
    remove_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > p
    sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))

    # Scatter back to original ordering
    return logits.scatter(dim=-1, index=sorted_indices, src=sorted_logits)
