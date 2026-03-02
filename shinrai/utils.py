import importlib.util
import subprocess
import sys
import os
import logging

# Configure logging once for the package
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


_checked_packages = set()

def ensure_dependencies():
    """Ensure all required packages are installed.

    Packages that are non-critical (like `cloudscraper`) are attempted once and
    marked as checked so that repeated imports of the package submodules don't
    keep trying to install them.  If the runtime environment prohibits pip
    installs (PEP‑668), the failure is logged but suppressed gracefully.
    """
    required_packages = {
        'torch': 'torch',
        'transformers': 'transformers',
        'sentence_transformers': 'sentence-transformers',
        'nltk': 'nltk',
        'sklearn': 'scikit-learn',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'networkx': 'networkx',
        'textblob': 'textblob',
        'spacy': 'spacy',
        'PyPDF2': 'PyPDF2',
        'pdfplumber': 'pdfplumber',
        # `cloudscraper` is optional and may not be installable in restricted
        # environments; we attempt it once only when first needed.
        'cloudscraper': 'cloudscraper',
    }

    for package_name, pip_name in required_packages.items():
        if package_name in _checked_packages:
            continue
        _checked_packages.add(package_name)

        if importlib.util.find_spec(package_name) is None:
            logger.info(f"[*] Installing {pip_name}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
                logger.info(f"[+] {pip_name} installed")
            except Exception as e:
                logger.warning(f"Could not install {pip_name}: {e}. "
                               "Please install it manually if you need the feature.")


def setup_nltk():
    """Download necessary NLTK data if not already present."""

    import nltk

    nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)

    for resource in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'vader_lexicon']:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
        except LookupError:
            nltk.download(resource, download_dir=nltk_data_dir, quiet=True)


import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
else:
    logger.info("Using CPU")
