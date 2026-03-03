"""Top-level package for Shinrai chatbot."""

from .utils import ensure_dependencies, setup_nltk, DEVICE, logger
from .memory import ConversationMemory
from .knowledge import KnowledgeGraph
from .image import ImageGenerator, IMAGE_SENTINEL
from .response import ResponseGenerator
from .scraper import WebScraper
from .core import Shinrai

# LLM components (optional — only exported if available)
try:
    from .llm_tokenizer import BPETokenizer
    from .llm_model import GPT, GPTConfig
    from .llm_trainer import LLMTrainer, TrainerConfig
    from .llm_generate import LLMGenerator
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False

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
    'ImageGenerator',
    'IMAGE_SENTINEL',
    'ResponseGenerator',
    'WebScraper',
    'Shinrai',
    # LLM
    'BPETokenizer',
    'GPT',
    'GPTConfig',
    'LLMTrainer',
    'TrainerConfig',
    'LLMGenerator',
]
