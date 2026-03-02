import argparse
import sys

from .core import Shinrai
from .utils import ensure_dependencies, setup_nltk


# prepare environment when package is executed as a module
ensure_dependencies()
setup_nltk()


def main():
    parser = argparse.ArgumentParser(description='Shinrai - Advanced Uncensored AI Chatbot')
    parser.add_argument('command', choices=['train', 'chat', 'interactive'],
                        help='Command to execute')
    parser.add_argument('--url', help='URL to start scraping from')
    parser.add_argument('--crawl', type=int, default=100,
                        help='Maximum number of pages to crawl')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for embedding creation (larger uses more GPU)')
    parser.add_argument('--model', default='shinrai_model',
                        help='Model directory path')
    parser.add_argument('--lazy', action='store_true',
                        help='Delay model initialization until first use (faster start)')
    parser.add_argument('--file', help='File to load for training (txt, pdf, json, jsonl)')
    parser.add_argument('--dir', help='Directory to load for training (looks for txt, pdf, jsonl)')
    parser.add_argument('--proxy', help='Optional HTTP/HTTPS proxy to use for web requests')
    parser.add_argument('--no-topics', action='store_true',
                        help='Do not retrain topic model (skips CPU-heavy step)')

    args = parser.parse_args()

    # Initialize chatbot (optionally lazily)
    shinrai = Shinrai(model_path=args.model, lazy=args.lazy)
    # apply proxy settings if provided
    if args.proxy:
        shinrai.web_scraper.proxies = {
            'http': args.proxy,
            'https': args.proxy
        }

    if args.command in ['train']:
        train_kwargs = {
            'embedding_batch_size': args.batch_size,
            'no_topics': args.no_topics
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

    elif args.command in ['chat', 'interactive']:
        shinrai.chat()


if __name__ == "__main__":
    main()
