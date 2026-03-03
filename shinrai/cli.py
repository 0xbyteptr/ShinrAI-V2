import argparse
import os
import sys

from .core import Shinrai
from .utils import ensure_dependencies, setup_nltk


# prepare environment when package is executed as a module
ensure_dependencies()
setup_nltk()


def main():
    parser = argparse.ArgumentParser(description='Shinrai - Advanced Uncensored AI Chatbot')
    parser.add_argument('command',
                        choices=['train', 'chat', 'interactive', 'llm-train', 'llm-chat'],
                        help='Command to execute')

    # ── retrieval-based training ────────────────────────────────────────────
    parser.add_argument('--url', help='URL to start scraping from')
    parser.add_argument('--crawl', type=int, default=100,
                        help='Maximum number of pages to crawl')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for embedding creation (larger uses more GPU)')
    parser.add_argument('--file', help='File to load for training (txt, pdf, json, jsonl)')
    parser.add_argument('--dir', help='Directory to load for training (looks for txt, pdf, jsonl)')
    parser.add_argument('--proxy', help='Optional HTTP/HTTPS proxy to use for web requests')
    parser.add_argument('--no-topics', action='store_true',
                        help='Do not retrain topic model (skips CPU-heavy step)')
    parser.add_argument('--hf-token', default=None,
                        help='Hugging Face access token for gated datasets '
                             '(falls back to HF_TOKEN env var)')

    # ── shared ──────────────────────────────────────────────────────────────
    parser.add_argument('--model', default='shinrai_model',
                        help='Model directory path')
    parser.add_argument('--lazy', action='store_true',
                        help='Delay model initialization until first use (faster start)')

    # ── LLM training ────────────────────────────────────────────────────────
    parser.add_argument('--vocab-size', type=int, default=16000,
                        help='BPE vocabulary size for the LLM tokenizer (default: 16000)')
    parser.add_argument('--llm-epochs', type=int, default=3,
                        help='Number of training epochs for the LLM (default: 3)')
    parser.add_argument('--llm-batch', type=int, default=8,
                        help='Batch size for LLM training windows (default: 8)')
    parser.add_argument('--llm-seq-len', type=int, default=512,
                        help='Context window length in tokens (default: 512)')
    parser.add_argument('--llm-lr', type=float, default=3e-4,
                        help='Peak learning rate for LLM AdamW (default: 3e-4)')
    parser.add_argument('--llm-size', choices=['small', 'medium'], default='small',
                        help='Model size: small ~45M params, medium ~125M params (default: small)')

    # ── LLM generation / chat ───────────────────────────────────────────────
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature for LLM generation (default: 0.8)')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Top-k token filter for LLM generation (default: 50)')
    parser.add_argument('--top-p', type=float, default=0.95,
                        help='Nucleus sampling threshold for LLM generation (default: 0.95)')
    parser.add_argument('--max-new-tokens', type=int, default=256,
                        help='Maximum tokens to generate per turn (default: 256)')
    parser.add_argument('--no-rag', action='store_true',
                        help='Disable retrieval-augmented context injection in llm-chat')

    args = parser.parse_args()

    # Initialize chatbot (optionally lazily)
    shinrai = Shinrai(model_path=args.model, lazy=args.lazy)
    # apply proxy settings if provided
    if args.proxy:
        shinrai.web_scraper.proxies = {
            'http': args.proxy,
            'https': args.proxy
        }

    # ── retrieval-based training ─────────────────────────────────────────────
    if args.command in ['train']:
        hf_token = args.hf_token or os.environ.get('HF_TOKEN') or None
        train_kwargs = {
            'embedding_batch_size': args.batch_size,
            'no_topics': args.no_topics,
            'hf_token': hf_token,
        }
        if args.url:
            shinrai.train(
                args.url,
                source_type='web',
                max_pages=args.crawl,
                **train_kwargs
            )
        elif args.file:
            shinrai.train(
                args.file,
                source_type='file',
                **train_kwargs
            )
        elif args.dir:
            shinrai.train(
                args.dir,
                source_type='directory',
                **train_kwargs
            )
        else:
            print("Please provide a data source (--url, --file, or --dir)")
            sys.exit(1)

    # ── retrieval-based chat ─────────────────────────────────────────────────
    elif args.command in ['chat', 'interactive']:
        shinrai.chat()

    # ── LLM training ─────────────────────────────────────────────────────────
    elif args.command == 'llm-train':
        shinrai.llm_train(
            vocab_size=args.vocab_size,
            epochs=args.llm_epochs,
            batch_size=args.llm_batch,
            seq_len=args.llm_seq_len,
            lr=args.llm_lr,
            model_size=args.llm_size,
        )

    # ── LLM chat ─────────────────────────────────────────────────────────────
    elif args.command == 'llm-chat':
        shinrai.llm_chat(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            use_rag=not args.no_rag,
        )


if __name__ == "__main__":
    main()
