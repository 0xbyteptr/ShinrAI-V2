"""Top-level package for Shinrai chatbot."""

from .utils import ensure_dependencies, setup_nltk, DEVICE, logger
from .memory import ConversationMemory
from .knowledge import KnowledgeGraph
from .response import ResponseGenerator
from .scraper import WebScraper
from .core import Shinrai

# Perform dependency check and data setup on import
ensure_dependencies()
setup_nltk()

__all__ = [
    'ensure_dependencies',
    'setup_nltk',
    'DEVICE',
    'logger',
    'ConversationMemory',
    'KnowledgeGraph',
    'ResponseGenerator',
    'WebScraper',
    'Shinrai',
]
