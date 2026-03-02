import argparse
import json
import os
import sys
import pickle
from pathlib import Path
from datetime import datetime
from typing import List
import re
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from .memory import ConversationMemory
from .knowledge import KnowledgeGraph
from .response import ResponseGenerator
from .scraper import WebScraper
from .utils import DEVICE, logger

class Shinrai:
    """Advanced uncensored AI chatbot"""

    def __init__(self, model_path: str = "shinrai_model", lazy: bool = False):
        """Create a Shinrai instance.

        Parameters
        ----------
        model_path:
            Directory (or .pt file) where the model is stored.
        lazy:
            If True, initialization skips loading the transformer and saved
            model data.  Models will be loaded automatically on first use.
            This can significantly reduce startup time when you only want to
            create the object without immediately chatting or training.
        """
        # if the user passes a path to a .pt file, use its parent directory
        p = Path(model_path)
        if p.suffix == '.pt':
            # ensure parent exists then remember model_file separately
            self.model_path = p.parent
            self.model_file = p
        else:
            self.model_path = p
            self.model_file = self.model_path / "shinrai_model.pt"
        self.model_path.mkdir(exist_ok=True)

        # Path where conversation interactions may be offloaded when RAM is
        # scarce.  Living inside the model directory keeps everything together.
        self.offload_file = str(self.model_path / "conversation_history.jsonl")

        # Core components
        self.documents = []
        self.document_metadata = []
        self.conversation_memory = ConversationMemory(offload_path=self.offload_file)
        self.knowledge_graph = KnowledgeGraph()
        self.response_generator = ResponseGenerator()
        self.web_scraper = WebScraper()

        # Embeddings and models (initialized lazily)
        self.embeddings = None
        self.transformer_model = None
        self.tokenizer = None
        self.tfidf_vectorizer = None
        self.topic_model = None

        # track whether we've performed expensive setup
        self._models_initialized = False
        self._model_data_loaded = False

        if not lazy:
            # regular behaviour: build everything now
            self._ensure_models()
            self.load_model()
            logger.info("Shinrai initialized successfully")
        else:
            logger.info("Shinrai initialized in lazy mode; models will load on demand")

    def _flatten_json(self, obj):
        """Recursively convert a JSON-like object into a single text string.

        Works with nested dictionaries and lists, producing space-separated
        output.  Used by both the JSON and JSONL loaders to avoid duplicating
        the same logic in multiple places.
        """
        if isinstance(obj, dict):
            return ' '.join(self._flatten_json(v) for v in obj.values())
        elif isinstance(obj, list):
            return ' '.join(self._flatten_json(v) for v in obj)
        else:
            return str(obj)

    def _initialize_models(self):
        """Initialize AI models.

        We try to load models in ``local_files_only`` mode first, which uses the
        HuggingFace cache and avoids network traffic.  If that fails (for
        instance the cache is empty) we fall back to a normal download.  Once
        the weights have been pulled they are stored in the cache and future
        instantiations – even in new processes – will not re‑download them.
        """
        # load the sentence transformer; failure shouldn't prevent tokenizer
        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        try:
            from sentence_transformers import SentenceTransformer

            load_kwargs = {}
            if os.environ.get('TRANSFORMERS_OFFLINE') or os.environ.get('HF_LOCAL_FILES_ONLY'):
                load_kwargs['local_files_only'] = True

            try:
                self.transformer_model = SentenceTransformer(model_name, **load_kwargs)
            except Exception as e:
                # if offline load fails because nothing is cached, try again online
                logger.debug(f"offline load failed ({e}), retrying online")
                self.transformer_model = SentenceTransformer(model_name)

            self.transformer_model = self.transformer_model.to(DEVICE)
            logger.info(f"Loaded transformer model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading sentence transformer: {e}")
            self.transformer_model = None

        # load tokenizer separately so that transformer failures don't block it
        try:
            from transformers import AutoTokenizer
            tok_kwargs = {}
            if os.environ.get('TRANSFORMERS_OFFLINE') or os.environ.get('HF_LOCAL_FILES_ONLY'):
                tok_kwargs['local_files_only'] = True
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', **tok_kwargs)
            logger.info("Loaded tokenizer: bert-base-uncased")
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            self.tokenizer = None

    def _ensure_models(self):
        """Make sure model components are initialized.

        This is idempotent and safe to call at the start of any operation that
        requires the transformer, tokenizer, or embeddings.  It is used by
        training and retrieval to lazily load resources when ``lazy`` mode is
        enabled.
        """
        if not self._models_initialized:
            logger.info("Lazy initialization of models...")
            self._initialize_models()
            # we may also want to load data if it wasn't already
            if not self._model_data_loaded:
                self.load_model()
            # only mark as initialized if the critical components loaded
            if self.transformer_model is not None and self.tokenizer is not None:
                self._models_initialized = True
            else:
                logger.warning("Model initialization incomplete; will retry on demand")

    def train(self, data_source: str, source_type: str = 'web', *, embedding_batch_size: int = 512, **kwargs):
        """Train the model from various data sources.

        Parameters
        ----------
        data_source:
            URL, filename, or directory to load documents from.
        source_type:
            One of ``'web'``, ``'file'`` or ``'directory'``.
        embedding_batch_size:
            Number of documents to encode in a single batch.  Larger values
            will consume more GPU memory but also increase utilisation during
            embedding creation; you can tune this if your GPU seems idle.
        """
        # ensure that models are ready before touching embeddings
        self._ensure_models()
        logger.info(f"Starting training; using device {DEVICE}")
        # Collect data
        if source_type == 'web':
            # reset scraper state so repeated calls fetch again
            try:
                self.web_scraper.visited_urls.clear()
                self.web_scraper.scraped_data.clear()
            except Exception:
                pass
            scraped_data = self.web_scraper.scrape(data_source, kwargs.get('max_pages', 100))
            texts = [item['content'] for item in scraped_data if item['content']]
            self.document_metadata.extend(scraped_data)
            
            if not scraped_data:
                logger.warning(
                    "Scraping returned no pages; check the start URL and network. "
                    "You might need to supply a proxy via --proxy or train from a local file."
                )
            if not texts:
                logger.warning("No text extracted from scraped pages")
                
        elif source_type == 'file':
            texts = self._load_from_file(data_source)
            if not texts:
                logger.warning(f"No text loaded from file {data_source}")
                
        elif source_type == 'directory':
            texts = self._load_from_directory(data_source)
            if not texts:
                logger.warning(f"No text loaded from directory {data_source}")
        else:
            raise ValueError(f"Unknown source type: {source_type}")

        # Process and store documents
        previous_count = len(self.documents)
        self.documents.extend(texts)
        logger.info(f"Added {len(texts)} new documents (had {previous_count} before, now {len(self.documents)})")

        if texts:
            # create embeddings for the new texts and append
            self._create_embeddings(texts, batch_size=embedding_batch_size)

            # build graph incrementally
            self._build_knowledge_graph(texts)

            # retrain topic model on all documents so it stays up‑to‑date
            if not kwargs.get('no_topics'):
                self._train_topic_model(self.documents)
            else:
                logger.info("Skipping topic model update (--no-topics)")

        # Save model
        self.save_model()
        logger.info(f"Training complete. Total documents: {len(self.documents)}")

    def _load_from_file(self, path: str) -> List[str]:
        """Load text data from a single file.

        Supports plain text, PDF and JSON documents by inspecting the file
        extension.  Other binary formats may be added later.
        """
        suffix = Path(path).suffix.lower()
        try:
            if suffix == '.txt':
                with open(path, 'r', encoding='utf-8') as f:
                    return [f.read()]

            elif suffix == '.pdf':
                try:
                    import pdfplumber
                except ImportError:
                    logger.warning('pdfplumber not installed, falling back to PyPDF2')
                    pdfplumber = None

                text = ''
                if pdfplumber:
                    with pdfplumber.open(path) as pdf:
                        for page in pdf.pages:
                            text += page.extract_text() or ''
                else:
                    # fallback to PyPDF2
                    from PyPDF2 import PdfReader
                    reader = PdfReader(path)
                    for page in reader.pages:
                        text += page.extract_text() or ''
                return [text]

            elif suffix == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    import json
                    data = json.load(f)
                # if it's the conversational schema we expect from social_skills.json
                if isinstance(data, dict) and 'conversations' in data:
                    texts = []
                    for conv in data.get('conversations', []):
                        pats = conv.get('patterns', [])
                        resps = conv.get('responses', [])
                        texts.append(' '.join(pats + resps))
                    return texts

                flat = self._flatten_json(data)
                return [flat] if flat else []

            elif suffix == '.jsonl':
                texts = []
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        import json
                        buffer = ''
                        for raw in f:
                            line = raw.strip()
                            if not line:
                                continue
                            # accumulate until we can parse a complete JSON object
                            buffer += line
                            try:
                                item = json.loads(buffer)
                                flat = self._flatten_json(item)
                                if flat:
                                    texts.append(flat)
                                buffer = ''
                            except json.JSONDecodeError as e:
                                # if error appears to be due to unexpected end
                                # of data, keep buffering; otherwise log and reset
                                msg = str(e)
                                if 'Expecting value' in msg or 'Unterminated' in msg or e.pos >= len(buffer) - 1:
                                    # not enough data yet, continue reading
                                    buffer += ' '
                                    continue
                                logger.warning(f"Skipping invalid JSONL record in {path}: {e}")
                                buffer = ''
                        # handle any leftover buffer
                        if buffer:
                            try:
                                item = json.loads(buffer)
                                flat = self._flatten_json(item)
                                if flat:
                                    texts.append(flat)
                            except json.JSONDecodeError:
                                logger.warning(f"Leftover invalid JSON in {path}: {buffer[:50]}")
                except Exception as e:
                    logger.error(f"Failed to parse JSONL {path}: {e}")
                return texts

            elif suffix == '.csv':
                # read CSV and concatenate each row's values into a single string
                try:
                    import csv
                    rows = []
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        reader = csv.DictReader(f)
                        for r in reader:
                            # join all fields with spaces
                            rows.append(' '.join(str(v) for v in r.values()))
                    return rows
                except Exception as e:
                    logger.error(f"Failed to parse CSV {path}: {e}")
                    return []

            elif suffix == '.bin':
                # load generic binary file and attempt to decode as text
                try:
                    with open(path, 'rb') as f:
                        data = f.read()
                    # try utf-8 then latin1 fallback, ignore errors
                    try:
                        text = data.decode('utf-8')
                    except Exception:
                        text = data.decode('latin1', errors='ignore')
                    return [text]
                except Exception as e:
                    logger.error(f"Failed to load binary file {path}: {e}")
                    return []

            else:
                # treat as generic text
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    return [f.read()]
        except Exception as e:
            logger.error(f"Failed to load file {path}: {e}")
            return []

    def _load_from_directory(self, path: str) -> List[str]:
        """Load text data from supported files within a directory"""
        texts = []
        base = Path(path)
        for file in base.rglob('*'):
            if file.suffix.lower() in ['.txt', '.pdf', '.json', '.jsonl', '.csv', '.bin']:
                try:
                    texts.extend(self._load_from_file(str(file)))
                except Exception as e:
                    logger.error(f"Failed to read {file}: {e}")
        return texts

    def _create_embeddings(self, texts: List[str], batch_size: int = 32):
        """Create or append embeddings for a batch of documents.

        ``batch_size`` can be tuned to make better use of GPU resources; on a
        large card you may be able to increase it to 128, 256 or even more.  We
        also pass the device explicitly to the encoder to avoid any ambiguity.
        """
        if not texts:
            return

        if self.transformer_model is None:
            logger.warning("Transformer model not available; skipping embedding creation")
            return

        logger.info(f"Creating document embeddings on {DEVICE} (batch size {batch_size})...")
        new_embs = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                # specify device so SentenceTransformer knows where to put tensors
                emb = self.transformer_model.encode(
                    batch,
                    convert_to_tensor=True,
                    device=str(DEVICE)
                )
                # ensure embeddings are on the desired device
                if emb.device != DEVICE:
                    emb = emb.to(DEVICE)
                new_embs.append(emb)
                if (i // batch_size) % 10 == 0:
                    logger.info(f"Processed {i + len(batch)}/{len(texts)} new documents")
            except Exception as e:
                logger.error(f"Error in batch {i}: {e}")

        if new_embs:
            new_tensor = torch.cat(new_embs, dim=0)
            # keep everything on DEVICE to avoid needless CPU-GPU transfers
            new_tensor = new_tensor.to(DEVICE)
            if self.embeddings is None:
                self.embeddings = new_tensor
            else:
                # existing embeddings should already live on DEVICE
                if self.embeddings.device != DEVICE:
                    self.embeddings = self.embeddings.to(DEVICE)
                self.embeddings = torch.cat([self.embeddings, new_tensor], dim=0)
            logger.info(f"Embeddings updated, shape now {self.embeddings.shape}")

    def _build_knowledge_graph(self, texts: List[str]):
        """Build knowledge graph from documents"""
        logger.info("Building knowledge graph...")
        
        # build graph over all supplied texts (size should be manageable)
        for i, text in enumerate(texts[:100]):  # Limit to first 100 docs for efficiency
            doc_id = f"doc_{i}"
            self.knowledge_graph.add_document(doc_id, text)
            
            if i and i % 20 == 0:
                logger.info(f"Processed {i}/{len(texts)} documents for knowledge graph")
                
        logger.info(f"Knowledge graph built with {self.knowledge_graph.graph.number_of_nodes()} nodes")

    def _clean_texts_for_topics(self, texts: List[str]) -> List[str]:
        """Return a cleaned copy of texts aimed at topic modeling.

        Removes purely numeric tokens, ISBN occurrences and other
        garbage that frequently appears in scraped wiki dumps.
        """
        cleaned = []
        for t in texts:
            # drop numbers and the word isbn
            t2 = re.sub(r"\b(?:\d+|isbn)\b", "", t, flags=re.I)
            cleaned.append(t2)
        return cleaned

    def _train_topic_model(self, texts: List[str]):
        """Train topic model on documents.

        By default this uses scikit-learn's LDA implementation, which is
        CPU-bound and may consume all cores for large corpora.  If the
        optional `cuml` package from RAPIDS is installed we will instead run
        the model on the GPU, which can significantly reduce CPU usage and
        make training overlap with the embedding step.
        """
        logger.info("Training topic model...")

        # clean the corpus before vectorizing in order to avoid junk topics
        safe_texts = self._clean_texts_for_topics(texts)

        # Create TF-IDF vectors
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2)
        )

        tfidf_matrix = self.tfidf_vectorizer.fit_transform(safe_texts)

        # Attempt to use GPU-accelerated LDA if available
        try:
            from cuml.decomposition import LatentDirichletAllocation as cuLDA
            gpu_available = True
        except ImportError:
            gpu_available = False

        if gpu_available:
            logger.info("cuML detected, training topic model on GPU")
            # cuML expects float32 arrays
            tfidf_gpu = tfidf_matrix.astype('float32')
            self.topic_model = cuLDA(
                n_components=20,
                random_state=42,
                max_iter=10
            )
            try:
                self.topic_model.fit(tfidf_gpu)
            except Exception as e:
                logger.warning(f"GPU topic model failed ({e}), falling back to CPU")
                self.topic_model = LatentDirichletAllocation(
                    n_components=20,
                    random_state=42,
                    max_iter=10
                )
                self.topic_model.fit(tfidf_matrix)
        else:
            self.topic_model = LatentDirichletAllocation(
                n_components=20,
                random_state=42,
                max_iter=10
            )
            self.topic_model.fit(tfidf_matrix)

        logger.info("Topic model trained")

    def chat(self):
        """Interactive chat session"""
        # make sure models and stored data are available before starting
        self._ensure_models()

        print("\n" + "=" * 70)
        print("🤖 Shinrai Advanced AI Chatbot - Uncensored")
        print("=" * 70)
        print(f"Device: {DEVICE}")
        print(f"Knowledge base: {len(self.documents)} documents")
        print("\nCommands:")
        print("  /help     - Show this help")
        print("  /memory   - Show conversation memory")
        print("  /graph    - Show knowledge graph stats")
        print("  /topics   - Show main topics")
        print("  /save     - Save conversation")
        print("  /clear    - Clear conversation history")
        print("  /exit     - Exit chat")
        print("=" * 70)

        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()

                # Handle commands
                if user_input.startswith('/'):
                    self._handle_command(user_input[1:])
                    continue

                if not user_input:
                    continue

                # Generate response
                logger.info(f"Processing query: {user_input[:50]}...")

                # Get relevant context
                relevant_docs = self._get_relevant_documents(user_input)

                # Generate response
                response = self.response_generator.generate(
                    user_input,
                    relevant_docs,
                    self.conversation_memory,
                    self.knowledge_graph
                )

                # Store in memory
                self.conversation_memory.add_interaction(user_input, response)

                # Print response
                print(f"\n🤖 Shinrai: {response}")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                # log full traceback to aid debugging
                logger.exception("Error in chat")
                print(f"\n❌ Error: {e}")

    def _handle_command(self, command: str):
        """Handle chat commands"""
        if command == 'help':
            print("\n📚 Available commands:")
            print("  /help     - Show this help")
            print("  /memory   - Show conversation memory")
            print("  /graph    - Show knowledge graph stats")
            print("  /topics   - Show main topics")
            print("  /save     - Save conversation")
            print("  /clear    - Clear conversation history")
            print("  /exit     - Exit chat")

        elif command == 'memory':
            summary = self.conversation_memory.get_summary()
            print("\n🧠 Conversation Memory:")
            print(f"  Interactions: {summary['total_interactions']}")
            print(f"  Duration: {summary['duration']}")
            print(f"  Avg sentiment: {summary['avg_sentiment']:.2f}")
            print(f"  Main topics: {', '.join(summary['main_topics'][:5])}")

        elif command == 'graph':
            print("\n📊 Knowledge Graph Stats:")
            print(f"  Nodes: {self.knowledge_graph.graph.number_of_nodes()}")
            print(f"  Edges: {self.knowledge_graph.graph.number_of_edges()}")

        elif command == 'topics':
            if self.topic_model and self.tfidf_vectorizer:
                feature_names = self.tfidf_vectorizer.get_feature_names_out()
                print("\n📈 Main Topics:")
                for topic_idx, topic in enumerate(self.topic_model.components_[:5]):
                    top_words = [feature_names[i] for i in topic.argsort()[:-10:-1]]
                    print(f"  Topic {topic_idx + 1}: {', '.join(top_words)}")

        elif command == 'save':
            self._save_conversation()
            print("✅ Conversation saved")

        elif command == 'clear':
            # keep using same offload file when resetting memory
            self.conversation_memory = ConversationMemory(offload_path=self.offload_file)
            print("🧹 Conversation history cleared")

        elif command == 'exit':
            print("👋 Goodbye!")
            sys.exit(0)

    def _get_relevant_documents(self, query: str, top_k: int = 5) -> List[str]:
        """Get most relevant documents for query"""
        # embeddings require models, so initialize if necessary
        self._ensure_models()
        # if the transformer model failed to load, skip retrieval entirely
        if self.transformer_model is None or self.embeddings is None:
            # fallback: return first few documents without any similarity ranking
            return self.documents[:top_k] if self.documents else []

        try:
            # Encode query
            query_embedding = self.transformer_model.encode([query], convert_to_tensor=True)
            query_embedding = query_embedding.to(DEVICE)
        except Exception as e:
            logger.error(f"Failed to encode query for retrieval: {e}")
            return self.documents[:top_k] if self.documents else []

        # Calculate similarities
        similarities = F.cosine_similarity(query_embedding, self.embeddings)

        # Get top-k indices
        top_scores, top_indices = torch.topk(similarities, min(top_k * 2, len(similarities)))

        # Filter by threshold (lowered for broader recall)
        threshold = 0.1
        relevant_docs = []

        for score, idx in zip(top_scores, top_indices):
            if score > threshold:
                relevant_docs.append(self.documents[idx])
            if len(relevant_docs) >= top_k:
                break

        return relevant_docs if relevant_docs else [self.documents[top_indices[0]]]

    def _save_conversation(self):
        """Save conversation history to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.model_path / f"conversation_{timestamp}.json"

        conversation_data = {
            'timestamp': timestamp,
            'history': list(self.conversation_memory.history),
            'summary': self.conversation_memory.get_summary()
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, ensure_ascii=False, indent=2)

    def save_model(self):
        """Save model to disk - FIXED VERSION"""
        logger.info("Saving model...")
        
        # Convert embeddings to CPU and ensure they're in a serializable format
        embeddings_cpu = None
        if self.embeddings is not None:
            # Detach and move to CPU
            embeddings_cpu = self.embeddings.detach().cpu()
        
        # Save TF-IDF vectorizer separately using pickle
        tfidf_path = self.model_path / "tfidf_vectorizer.pkl"
        if self.tfidf_vectorizer is not None:
            try:
                with open(tfidf_path, 'wb') as f:
                    pickle.dump(self.tfidf_vectorizer, f)
                logger.info(f"Saved TF-IDF vectorizer to {tfidf_path}")
            except Exception as e:
                logger.error(f"Failed to save TF-IDF vectorizer: {e}")
        
        # Save topic model separately
        topic_path = self.model_path / "topic_model.pkl"
        if self.topic_model is not None:
            try:
                with open(topic_path, 'wb') as f:
                    pickle.dump(self.topic_model, f)
                logger.info(f"Saved topic model to {topic_path}")
            except Exception as e:
                logger.error(f"Failed to save topic model: {e}")
        
        # Save knowledge graph separately
        kg_path = self.model_path / "knowledge_graph.pkl"
        if self.knowledge_graph is not None:
            try:
                with open(kg_path, 'wb') as f:
                    pickle.dump(self.knowledge_graph, f)
                logger.info(f"Saved knowledge graph to {kg_path}")
            except Exception as e:
                logger.error(f"Failed to save knowledge graph: {e}")
        
        # Prepare data for torch.save - only include basic Python types and tensors
        model_data = {
            'documents': self.documents,
            'document_metadata': self.document_metadata,
            'embeddings': embeddings_cpu,
            'metadata': {
                'num_documents': len(self.documents),
                'last_trained': datetime.now().isoformat(),
                'model_version': '3.0',
                'device': str(DEVICE)
            }
        }

        # Save to file using torch.save with additional error handling
        model_file = self.model_path / "shinrai_model.pt"
        try:
            # Save with pickle protocol 4 for better compatibility
            torch.save(model_data, model_file, pickle_protocol=4)
            logger.info(f"Model saved to {model_file}")
            
            # Verify the file was saved correctly
            if model_file.exists():
                file_size = model_file.stat().st_size
                logger.info(f"Model file size: {file_size / 1024:.2f} KB")
            else:
                logger.error("Model file was not created!")
                
        except Exception as e:
            logger.error(f"Failed to save model with torch.save: {e}")
            
            # Fallback: try saving with pickle directly
            try:
                logger.info("Attempting fallback save with pickle...")
                fallback_path = self.model_path / "shinrai_model_fallback.pkl"
                with open(fallback_path, 'wb') as f:
                    pickle.dump(model_data, f, protocol=4)
                logger.info(f"Model saved with pickle fallback to {fallback_path}")
            except Exception as e2:
                logger.error(f"Fallback save also failed: {e2}")

    def load_model(self) -> bool:
        """Load model from disk - FIXED VERSION

        This method can be called lazily when the data is first needed.  It
        avoids doing expensive work during initialization if the user only
        wanted to instantiate the object but not immediately chat or train.
        """
        if self._model_data_loaded:
            return True
        model_file = self.model_path / "shinrai_model.pt"
        fallback_file = self.model_path / "shinrai_model_fallback.pkl"
        
        # Try primary model file first
        if model_file.exists():
            try:
                logger.info(f"Loading model from {model_file}")
                # Use weights_only=False for full compatibility
                model_data = torch.load(model_file, map_location='cpu', weights_only=False)
                
                self.documents = model_data['documents']
                self.document_metadata = model_data.get('document_metadata', [])
                
                if model_data['embeddings'] is not None:
                    self.embeddings = model_data['embeddings'].to(DEVICE)
                    logger.info(f"Loaded embeddings with shape {self.embeddings.shape}")
                
                logger.info(f"Model loaded from {model_file}")
                logger.info(f"Contains {len(self.documents)} documents")
                
            except Exception as e:
                logger.error(f"Failed to load model from {model_file}: {e}")
                # Try fallback file
                if fallback_file.exists():
                    return self._load_fallback_model(fallback_file)
                else:
                    return False
        
        # Try fallback file if primary doesn't exist
        elif fallback_file.exists():
            return self._load_fallback_model(fallback_file)
        
        else:
            logger.info("No existing model found")
            return False
        
        # Load additional components
        self._load_additional_components()
        
        return True

    def _load_fallback_model(self, fallback_file: Path) -> bool:
        """Load model from pickle fallback file"""
        try:
            logger.info(f"Loading model from fallback file {fallback_file}")
            with open(fallback_file, 'rb') as f:
                model_data = pickle.load(f)
            
            self.documents = model_data['documents']
            self.document_metadata = model_data.get('document_metadata', [])
            
            if model_data['embeddings'] is not None:
                self.embeddings = model_data['embeddings'].to(DEVICE)
                logger.info(f"Loaded embeddings with shape {self.embeddings.shape}")
            
            logger.info(f"Model loaded from fallback file")
            logger.info(f"Contains {len(self.documents)} documents")
            
            self._load_additional_components()
            return True
            
        except Exception as e:
            logger.error(f"Failed to load fallback model: {e}")
            return False

    def _load_additional_components(self):
        """Load additional components (TF-IDF, topic model, knowledge graph)"""
        # Load TF-IDF vectorizer if exists
        self._model_data_loaded = True
        tfidf_path = self.model_path / "tfidf_vectorizer.pkl"
        if tfidf_path.exists():
            try:
                with open(tfidf_path, 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
                logger.info("Loaded TF-IDF vectorizer")
            except Exception as e:
                logger.error(f"Failed to load TF-IDF vectorizer: {e}")
        
        # Load topic model if exists
        topic_path = self.model_path / "topic_model.pkl"
        if topic_path.exists():
            try:
                with open(topic_path, 'rb') as f:
                    self.topic_model = pickle.load(f)
                logger.info("Loaded topic model")
            except Exception as e:
                logger.error(f"Failed to load topic model: {e}")
        
        # Load knowledge graph if exists
        kg_path = self.model_path / "knowledge_graph.pkl"
        if kg_path.exists():
            try:
                with open(kg_path, 'rb') as f:
                    self.knowledge_graph = pickle.load(f)
                logger.info("Loaded knowledge graph")
            except Exception as e:
                logger.error(f"Failed to load knowledge graph: {e}")
        elif self.documents:
            # Rebuild knowledge graph from documents if not saved
            try:
                logger.info("Rebuilding knowledge graph from documents...")
                self._build_knowledge_graph(self.documents[:100])  # Limit for efficiency
                logger.info("Knowledge graph rebuilt from documents")
            except Exception as e:
                logger.error(f"Failed to rebuild knowledge graph: {e}")