from collections import defaultdict
from typing import List, Tuple

import networkx as nx
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


class KnowledgeGraph:
    """Builds and maintains a knowledge graph from training data"""

    def __init__(self):
        self.graph = nx.Graph()
        self.entity_embeddings = {}
        self.relation_weights = defaultdict(float)

    def add_document(self, doc_id: str, text: str, entities: List[str] = None):
        """Add document to knowledge graph"""
        if entities is None:
            # Extract entities using simple heuristics
            entities = self._extract_entities(text)

        # Add document node
        self.graph.add_node(doc_id, type='document', text=text[:100])

        # Add entity nodes and connections
        for entity in entities:
            if not self.graph.has_node(entity):
                self.graph.add_node(entity, type='entity', occurrences=1)
            else:
                self.graph.nodes[entity]['occurrences'] = self.graph.nodes[entity].get('occurrences', 0) + 1

            # Add edge between document and entity
            if self.graph.has_edge(doc_id, entity):
                self.graph[doc_id][entity]['weight'] += 1
            else:
                self.graph.add_edge(doc_id, entity, weight=1)

    def _extract_entities(self, text: str) -> List[str]:
        """Extract potential entities from text"""
        words = word_tokenize(text)
        entities = []

        for i, word in enumerate(words):
            # Check for proper nouns (capitalized)
            if word and word[0].isupper() and len(word) > 1 and word.lower() not in stopwords.words('english'):
                # Check if it's part of a multi-word entity
                if i < len(words) - 1 and words[i + 1][0].isupper():
                    entities.append(' '.join(words[i:i + 2]))
                else:
                    entities.append(word)

        return list(set(entities))[:20]  # Limit entities per document

    def find_related(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find entities/documents related to query"""
        query_entities = self._extract_entities(query)

        if not query_entities:
            return []

        # Calculate relevance scores
        scores = defaultdict(float)

        for entity in query_entities:
            if self.graph.has_node(entity):
                # Add direct entity matches
                scores[entity] += 1.0

                # Add connected documents
                for neighbor in self.graph.neighbors(entity):
                    if self.graph.nodes[neighbor].get('type') == 'document':
                        weight = self.graph[entity][neighbor].get('weight', 1)
                        scores[neighbor] += weight * 0.5

        # Sort by score
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_items[:top_k]
