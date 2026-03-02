from collections import defaultdict, deque
from datetime import datetime
from typing import List, Dict, Optional
import json

import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob

# psutil is optional; only used if available for checking system memory.
try:
    import psutil
except ImportError:
    psutil = None


class ConversationMemory:
    """Manages conversation history and context"""

    def __init__(self, max_history: int = 50, max_tokens: int = 2000, offload_path: Optional[str] = None):
        """Create a conversation memory object.

        Parameters
        ----------
        max_history:
            Maximum number of interactions to keep in RAM before offloading.
        max_tokens:
            Rough token limit used for context window management.
        offload_path:
            Optional filesystem path where interactions will be appended when
            memory pressure is detected. If ``None`` no offloading will be
            performed.
        """
        self.max_history = max_history
        self.max_tokens = max_tokens
        self.offload_path = offload_path

        self.history = deque(maxlen=max_history)
        self.topics = deque(maxlen=10)
        self.sentiment_history = deque(maxlen=20)
        self.entities = defaultdict(set)
        self.context_window = {}
        self.last_interaction = datetime.now()

    def add_interaction(self, user_input: str, bot_response: str, metadata: Dict = None):
        """Add an interaction to memory"""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'user': user_input,
            'bot': bot_response,
            'metadata': metadata or {},
            'sentiment': self._analyze_sentiment(user_input + ' ' + bot_response),
            'topics': self._extract_topics(user_input + ' ' + bot_response),
            'entities': self._extract_entities(user_input + ' ' + bot_response)
        }

        self.history.append(interaction)
        self.sentiment_history.append(interaction['sentiment'])

        # Update topics
        for topic in interaction['topics']:
            self.topics.append(topic)

        # Update entities
        for entity_type, entities in interaction['entities'].items():
            self.entities[entity_type].update(entities)

        self.last_interaction = datetime.now()

        # If memory is constrained either by configured history length or
        # by system RAM, attempt to offload the oldest interactions to disk.
        self._maybe_offload()

    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text"""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except Exception:
            return 0.0

    def _extract_topics(self, text: str, num_topics: int = 3) -> List[str]:
        """Extract main topics from text"""
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalnum() and w not in stopwords.words('english')]
        # return most common words as simple topics
        from collections import Counter
        return [w for w, _ in Counter(words).most_common(num_topics)]

    def _extract_entities(self, text: str) -> Dict[str, set]:
        """Simple entity extraction"""
        entities = defaultdict(set)

        # Extract potential entities (capitalized words)
        words = word_tokenize(text)
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 1:
                # Check if it's part of a multi-word entity
                if i < len(words) - 1 and words[i + 1][0].isupper():
                    entity = ' '.join(words[i:i + 2])
                    entities['PERSON' if entity[0].isupper() else 'ORG'].add(entity)
                else:
                    entities['PERSON' if word[0].isupper() else 'ORG'].add(word)

        return dict(entities)

    def get_context(self, query: str, max_messages: int = 5) -> str:
        """Get relevant context from conversation history"""
        if not self.history:
            return ""

        # Get recent interactions
        recent = list(self.history)[-max_messages:]

        # Format context
        context_parts = []
        for interaction in recent:
            context_parts.append(f"User: {interaction['user']}")
            context_parts.append(f"Assistant: {interaction['bot']}")

        return "\n".join(context_parts)

    # --------- offloading helpers -----------------------------------------

    def _maybe_offload(self) -> None:
        """Decide whether to write items to disk and pop them from memory."""
        if not self.offload_path:
            return

        trigger = False
        # check deque length
        if len(self.history) >= self.max_history:
            trigger = True
        # optionally check available RAM if psutil is installed
        if psutil:
            avail = psutil.virtual_memory().available
            # offload if less than 200MB free
            if avail < 200 * 1024 * 1024:
                trigger = True

        if trigger:
            self._offload_oldest(count=1)

    def _offload_oldest(self, count: int = 1) -> None:
        """Write the oldest *count* interactions to the offload file.

        The interactions are appended as JSON lines.  We pop them from the
        left side of the deque so that memory usage is reclaimed.
        """
        if not self.offload_path or not self.history:
            return

        try:
            with open(self.offload_path, "a", encoding="utf-8") as f:
                for _ in range(count):
                    if not self.history:
                        break
                    item = self.history.popleft()
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        except Exception:
            # we deliberately swallow errors so that offloading never raises
            # during normal chat operations.
            pass

    def load_offloaded(self, max_lines: int = 100) -> List[Dict]:
        """Read up to *max_lines* of history from the offload file.

        This can be used when constructing context if the in‑memory history is
        empty or insufficient.  It does **not** remove the lines from disk.
        If the file doesn't exist or is unreadable an empty list is returned.
        """
        if not self.offload_path:
            return []

        result = []
        try:
            with open(self.offload_path, "r", encoding="utf-8") as f:
                for _ in range(max_lines):
                    line = f.readline()
                    if not line:
                        break
                    result.append(json.loads(line))
        except Exception:
            return []

        return result

    def get_summary(self) -> Dict:
        """Get summary of conversation memory"""
        return {
            'total_interactions': len(self.history),
            'duration': str(datetime.now() - self.last_interaction),
            'avg_sentiment': np.mean(self.sentiment_history) if self.sentiment_history else 0,
            'main_topics': list(set(self.topics)),
            'entities': {k: list(v)[:5] for k, v in self.entities.items()}
        }
