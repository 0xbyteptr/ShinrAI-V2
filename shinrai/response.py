import random
import re
from typing import List, Dict, Optional
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize


class ResponseGenerator:
    """Generates responses using multiple strategies with social intelligence"""
    
    def __init__(self):
        self.response_templates = self._load_templates()
        self.response_cache = {}
        self.conversation_state = {
            'last_interaction': None,
            'user_name': None,
            'mood': 'neutral',
            'interaction_count': 0,
            'topics_discussed': []
        }
        # Humanization helpers: fillers, colloquialisms, contractions
        self.filler_words = ["well", "um", "you know", "like", "actually", "honestly"]
        self.contractions = {
            " do not ": " don't ",
            " cannot ": " can't ",
            " I am ": " I'm ",
            " it is ": " it's ",
            " that is ": " that's ",
            " is not ": " isn't ",
            " are not ": " aren't ",
            " did not ": " didn't ",
            " does not ": " doesn't "
        }
        
    def _load_templates(self) -> Dict:
        """Load response templates with social responses"""
        return {
            # Greetings
            'greeting': [
                "Hello! How can I help you today?",
                "Hi there! What would you like to know?",
                "Greetings! I'm here to assist you.",
                "Hey! Ready to explore some knowledge?",
                "Hello! Nice to see you again."
            ],
            
            # Farewell
            'farewell': [
                "Goodbye! Feel free to come back anytime.",
                "Take care! Let me know if you need anything else.",
                "Until next time! Have a great day.",
                "Bye! It was nice chatting with you.",
                "See you later! Keep learning!"
            ],
            
            # Social queries
            'how_are_you': [
                "I'm doing great, thanks for asking! How about you?",
                "I'm functioning perfectly! How are you today?",
                "All systems operational and ready to help! How are you?",
                "I'm wonderful, thanks! What's on your mind?",
                "Doing well! I appreciate you asking. How are things with you?"
            ],
            
            'how_are_you_response': [
                "That's great to hear!",
                "I'm glad to hear that!",
                "Thanks for sharing!",
                "Good to know!",
                "Awesome!"
            ],
            
            'what_can_you_do': [
                "I can answer questions based on what I've learned from websites you train me on. Try asking me about topics from Wikipedia or other sources!",
                "I'm a knowledge chatbot - I can provide information on topics you've trained me on. What would you like to know?",
                "I can help you explore information from the web. Just train me on a topic and ask away!",
                "I'm designed to share knowledge from websites you've scraped. Ask me anything about what I've learned!"
            ],
            
            'who_are_you': [
                "I'm Shinrai, an uncensored AI chatbot designed to share knowledge from websites you train me on.",
                "I'm Shinrai - your personal knowledge assistant. I learn from websites you provide and can answer questions based on that information.",
                "I'm Shinrai, a customizable AI that learns from web content. You can train me on any topic!"
            ],
            
            'thanks': [
                "You're welcome! Happy to help.",
                "My pleasure!",
                "Anytime!",
                "Glad I could assist!",
                "You're welcome! Feel free to ask more questions."
            ],

            'quote': [
                "The best way to predict the future is to create it.",
                "Progress is built one small step at a time.",
                "Learning never exhausts the mind.",
                "Simplicity is the soul of efficiency.",
                "Code is read far more often than it is written."
            ],
            
            # Acknowledgment
            'acknowledgment': [
                "I understand.",
                "I see.",
                "Got it.",
                "Interesting point.",
                "That's a good observation.",
                "Thanks for sharing that."
            ],
            
            # Clarification
            'clarification': [
                "Could you please elaborate on that?",
                "I'm not sure I follow. Can you explain more?",
                "Could you provide more context?",
                "What specifically would you like to know about that?",
                "Can you rephrase that? I want to make sure I understand."
            ],
            
            # Uncertainty
            'uncertain': [
                "I'm not entirely sure about that.",
                "I don't have enough information to answer that confidently.",
                "That's outside my current knowledge base.",
                "I haven't learned about that yet. Try training me on that topic!",
                "I'm not sure I can answer that accurately. Would you like to ask something else?"
            ],
            
            # Opinion introductions
            'opinion': [
                "Based on my understanding, ",
                "From what I've learned, ",
                "According to the information available, ",
                "Here's what I know about that: ",
                "Let me share what I've learned: ",
                "The information I have indicates that "
            ],
            
            # Follow-ups
            'follow_up': [
                "Would you like to know more about that?",
                "Is there anything specific you'd like to know?",
                "Does that help answer your question?",
                "Let me know if you need more details.",
                "I can tell you more if you're interested."
            ]
        }
    
    def generate(self, query: str, context: List[str], 
                 conversation_memory, knowledge_graph) -> str:
        """Generate response using multiple strategies"""
        
        # Update conversation state
        self.conversation_state['interaction_count'] += 1
        self.conversation_state['last_interaction'] = datetime.now()
        
        # Check cache first
        import hashlib
        cache_key = hashlib.md5(f"{query}_{len(context)}_{self.conversation_state['interaction_count']}".encode()).hexdigest()
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        # Clean the query
        clean_query = query.lower().strip()
        
        # Check for social or special queries first (including summary requests)
        social_response = self._handle_social_query(clean_query, knowledge_graph, conversation_memory, context)
        if social_response:
            self.response_cache[cache_key] = social_response
            return social_response

        # if we have no useful context, try Wikipedia as a fallback for
        # definitional queries or when the retrieved documents look like
        # JSON/data dumps that won't produce good natural language answers.
        if self._is_unhelpful_context(context):
            wiki_text = self._fetch_wikipedia_summary(clean_query)
            if wiki_text:
                # return the summary directly (skip further processing)
                self.response_cache[cache_key] = wiki_text
                return wiki_text
        
        # Determine response type for knowledge queries
        response_type = self._classify_query(clean_query)
        
        # Generate response based on type
        if response_type == 'greeting':
            response = random.choice(self.response_templates['greeting'])
        elif response_type == 'farewell':
            response = random.choice(self.response_templates['farewell'])
        elif response_type == 'question':
            response = self._generate_question_response(clean_query, context)
        elif response_type == 'statement':
            response = self._generate_statement_response(clean_query, context, conversation_memory)
        else:
            response = self._generate_factual_response(clean_query, context)
        
        # Add personality based on conversation context
        response = self._add_personality(response, conversation_memory)
        
        # Add follow-up occasionally (20% chance)
        if self.conversation_state['interaction_count'] > 2 and random.random() < 0.2:
            follow_up = random.choice(self.response_templates['follow_up'])
            response = response + " " + follow_up
        
        # Humanize the text to sound more natural
        response = self._humanize_response(response)

        # Cache response
        self.response_cache[cache_key] = response
        
        return response
    
    def _handle_social_query(self, query: str, knowledge_graph=None, conversation_memory=None, context: List[str]=None) -> Optional[str]:
        """Handle social or special queries that don't need the usual knowledge search.

        Some questions like "summarize your knowledge" are routed here so we can
        give a higher‑level answer rather than blindly returning a slice of the
        training data.  The handler is intentionally generous about what it
        considers a social/special query so that they bypass the normal
        retrieval/response pipeline.

        ``context`` is the list of documents retrieved for the last query; it
        may be used when responding to provenance questions such as "how do you
        know that".
        """
        # Check for a request to summarise the bot's knowledge
        if 'summarize' in query and 'knowledge' in query:
            return self._summarize_knowledge(knowledge_graph)

        # Quote requests (including common typo: "quoute") should bypass
        # normal retrieval to avoid returning random corpus questions.
        if re.search(r'\b(quote|quoute|quotation)\b', query):
            if any(token in query for token in ['random', 'give me', 'tell me', 'provide', 'send']):
                return self._get_random_quote(context)

        # Topic summary requests (e.g. "summarize X") should not fall through
        # to greeting/farewell detection.
        if re.match(r'^summari[sz]e\b', query):
            if context:
                cleaned_context = self._clean_text(context[0])
                key_sentences = self._extract_key_sentences(cleaned_context, query, max_sentences=4)
                if key_sentences:
                    return " ".join(key_sentences)

                words = cleaned_context.split()
                if words:
                    return " ".join(words[:120]) + ("..." if len(words) > 120 else "")

            return random.choice(self.response_templates['uncertain'])

        if 'what do you know' in query or 'what have you learned' in query:
            return self._summarize_knowledge(knowledge_graph)

        # How are you queries
        if any(phrase in query for phrase in ['how are you', 'how do you do', 'how are things', 
                                              'how\'s it going', 'how is it going']):
            # Check if this is a response to "how are you"
            if self.conversation_state.get('last_was_how_are_you', False):
                self.conversation_state['last_was_how_are_you'] = False
                return random.choice(self.response_templates['how_are_you_response'])
            else:
                self.conversation_state['last_was_how_are_you'] = True
                return random.choice(self.response_templates['how_are_you'])
        
        # What can you do queries
        if any(phrase in query for phrase in ['what can you do', 'what do you do', 'what are you capable of',
                                              'what are your capabilities']):
            return random.choice(self.response_templates['what_can_you_do'])
        
        # Name / identity queries
        # include common variants so "what is your name" is treated as a social question
        if ('what is your name' in query or
            any(phrase in query for phrase in ['who are you', 'what are you', 'tell me about yourself'])):
            return random.choice(self.response_templates['who_are_you'])
        
        # Thanks queries
        if any(phrase in query for phrase in ['thank', 'thanks', 'appreciate it']):
            return random.choice(self.response_templates['thanks'])

        # Provenance or "how do you know" questions
        if 'how do you know' in query or 'where did you learn' in query or 'how can you say' in query:
            if context:
                snippet = context[0]
                if len(snippet) > 300:
                    snippet = snippet[:300] + '...'
                return (
                    "I base my answers on the information I've been trained on. "
                    f"For example, I found this in my documents: {snippet}"
                )
            return (
                "I know that because I search through the content I've learned "
                "whenever you ask a question. The response comes from those documents."
            )
        
        # Code generation / sample requests
        # e.g. "make a simple python program" or "write a python script"
        if 'python' in query and any(word in query for word in ['program', 'code', 'script', 'example', 'sample']):
            # give a tiny self-contained snippet rather than trying to search through
            # the knowledge base or fallback to Wikipedia
            return (
                "Here's a simple Python program you can try:\n\n"
                "```python\n"
                "# simple hello world example\n"
                "print('Hello, world!')\n"
                "```\n\n"
                "Save it to a file (e.g. `hello.py`) and run `python hello.py` to see the output."
            )
        
        return None

    def _get_random_quote(self, context: List[str]) -> str:
        """Return a short quote-like sentence from context or fallback list."""
        candidates = []
        if context:
            for doc in context[:3]:
                for sentence in sent_tokenize(doc):
                    cleaned = self._clean_text(sentence).strip()
                    if len(cleaned) < 35 or len(cleaned) > 180:
                        continue
                    if '?' in cleaned or '`' in cleaned:
                        continue
                    alpha_chars = sum(ch.isalpha() for ch in cleaned)
                    if alpha_chars < max(20, int(len(cleaned) * 0.6)):
                        continue
                    candidates.append(cleaned)

        if candidates:
            quote = random.choice(candidates)
            return f'"{quote}"'

        return f'"{random.choice(self.response_templates["quote"])}"'

    def _is_unhelpful_context(self, context: List[str]) -> bool:
        """Return True if the provided context is empty or looks like JSON/data.

        We don't want to generate answers from raw list/dict representations
        since they often originate from training on structured files and
        produce nonsense responses. Instead we fall back to an external
        source such as Wikipedia.
        """
        if not context:
            return True

        # if every context item starts with a bracket/brace or contains a
        # lot of punctuation typical of Python repr, treat as unhelpful.
        for ctx in context:
            text = ctx.strip()
            if text.startswith(('{', '[')):
                return True
            # simple heuristic: if there are more curly braces than words
            if text.count('{') + text.count('}') > len(text.split()) // 2:
                return True
            # also treat short context that looks like a word followed by a CEFR code
            # e.g. "crack B2" or "first name A2" as unhelpful
            if re.match(r'^[A-Za-z ]+\s+[A-Z]\d$', text):
                return True
        return False

    def _fetch_wikipedia_summary(self, query: str) -> Optional[str]:
        """Attempt to fetch a brief summary of *query* from Wikipedia.

        Uses the public REST API; returns plain text or ``None`` on failure.
        We try to strip off a variety of natural-language prefixes so that
        requests like "give me some facts about Poland" map to the topic
        "Poland" instead of the entire string.
        """
        # normalize whitespace and lower-case for prefix stripping
        q = query.strip().lower()
        # remove a set of common leading phrases
        q = re.sub(r'^(what is|who is|tell me about|'  # existing patterns
                   r'give me (some )?(facts?|information)( about)?|'  # added
                   r'provide (information|facts)( about)?|'  # added
                   r'show me (some )?(facts?|information)( about)?|'  # added
                   r'tell me (some )?(facts?|information)( about)?|'  # added
                   r'risks?|dangers?|effects?|benefits?|uses?|side effects?)\s+',
                   '', q, flags=re.I)
        # also strip common post‑prefix verbs to arrive at the core topic
        q = re.sub(r'^(of|in|to|taking|using|for)\s+', '', q)
        # strip punctuation and trailing question marks
        topic = re.sub(r"[^\w\s]", '', q)
        topic = topic.strip().replace(' ', '_')
        if not topic:
            return None
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
        try:
            import requests
            headers = {'User-Agent': 'ShinraiBot/1.0 (https://shinrai.wtf/)'}
            resp = requests.get(url, timeout=5, headers=headers)
            if resp.ok:
                data = resp.json()
                return data.get('extract')
        except Exception:
            pass
        return None

    def _summarize_knowledge(self, knowledge_graph) -> str:
        """Produce a brief summary of what the bot has learned.

        We look at the knowledge graph to determine the most common entities the
        model has seen.  Fallback to a generic message if the graph is empty.
        """
        if knowledge_graph is None:
            return "I don't have any knowledge yet. Train me on some websites to get started!"

        try:
            graph = knowledge_graph.graph
            if graph.number_of_nodes() == 0:
                return "I haven't been trained on anything yet. Give me a URL or file to learn from!"

            # gather entity nodes sorted by degree
            entities = [n for n, d in graph.nodes(data=True) if d.get('type') == 'entity']
            if entities:
                entities_sorted = sorted(entities, key=lambda e: graph.degree(e), reverse=True)[:6]
                return (
                    "I've learned about topics such as " + ", ".join(entities_sorted) +
                    ", among others. Ask me something more specific!"
                )
        except Exception:
            pass

        return "I have some information stored, but I'm not sure how to summarise it yet."
    
    def _classify_query(self, query: str) -> str:
        """Classify the type of query

        We handle not only traditional interrogatives but also imperatives that
        request information ("give me facts", "tell me about" etc.).  Without
        that the chatbot will treat polite commands as statements and respond in
       appropriately with a vague acknowledgment.
        """

        # Greeting detection (word/phrase boundaries to avoid false matches,
        # e.g. "anarhist" previously matched "hi").
        if re.search(r'\b(hello|hi|hey|greetings)\b', query) or re.search(r'\bgood\s+(morning|afternoon)\b', query):
            return 'greeting'

        # Farewell detection with boundaries
        if re.search(r'\b(bye|goodbye|exit|quit)\b', query) or re.search(r'\bsee\s+you\b', query):
            return 'farewell'

        # Question detection (standard interrogatives)
        if query.startswith(('what', 'why', 'how', 'when', 'where', 'who', 'which', 'can you', 'could you')):
            return 'question'
        if '?' in query:
            return 'question'

        # Questions phrased as nouns ("risks of…", "benefits of…", "effects of…").
        if re.match(r'^(risks?|dangers?|effects?|benefits?|uses?|side effects?)\b', query):
            return 'question'

        # Imperative forms that seek facts or information should also be treated
        # as questions so they invoke the more appropriate response logic.
        if re.match(r'^(give|tell|show|provide|list|make|write)\b', query):
            # optionally require a clue word to avoid false positives
            if any(word in query for word in ['fact', 'facts', 'information', 'about', 'details',
                                              'program', 'code', 'script', 'example', 'sample']):
                return 'question'

        # Statement detection
        if len(query.split()) > 5:
            return 'statement'

        # Fallback to factual answer for short utterances
        return 'factual'
    
    def _generate_question_response(self, query: str, context: List[str]) -> str:
        """Generate response for questions with better formatting"""
        # if the supplied context isn't useful we may still be able to answer via
        # Wikipedia.  This mirrors the check performed earlier in ``generate``
        # but covers the case where classification changed the response path.
        if self._is_unhelpful_context(context):
            wiki = self._fetch_wikipedia_summary(query)
            if wiki:
                return wiki

        if not context:
            return random.choice(self.response_templates['uncertain'])

        # Combine top candidate contexts to reduce single-document bias.
        cleaned_contexts = [self._clean_text(ctx) for ctx in context[:3]]
        merged = " ".join(cleaned_contexts)

        # treat trivial dictionary-like entries (e.g. "crack B2" or "first name A2") as unhelpful
        if re.match(r'^[A-Za-z ]+\s+[A-Z]\d$', merged):
            return random.choice(self.response_templates['uncertain'])

        # Extract key sentences related to the question
        key_sentences = self._extract_key_sentences(merged, query, max_sentences=3)
        
        if key_sentences:
            # Join sentences with proper spacing
            response_text = ' '.join(key_sentences)
        else:
            # Take first part of the context
            words = merged.split()
            if len(words) > 100:
                response_text = ' '.join(words[:100]) + "..."
            else:
                response_text = merged
        
        # Remove duplicates
        sentences = response_text.split('. ')
        unique_sentences = []
        seen = set()
        for sent in sentences:
            if sent not in seen:
                unique_sentences.append(sent)
                seen.add(sent)
        
        response_text = '. '.join(unique_sentences)
        
        # Choose an appropriate intro
        if "what is" in query or query.startswith("define "):
            intro = random.choice([
                f"Here's the definition: ",
                f"According to what I know, "
            ])
        elif "how to" in query:
            intro = random.choice([
                f"Here's how: ",
                f"You can do that by ",
                f"The process involves "
            ])
        elif "why" in query:
            intro = random.choice([
                f"The reason is ",
                f"This happens because ",
                f"Here's why: "
            ])
        else:
            intro = random.choice(self.response_templates['opinion'])
        
        # Format the response
        response = intro + response_text
        
        # Ensure proper punctuation
        if response and response[-1] not in '.!?':
            response += '.'
        
        return response
    
    def _generate_statement_response(self, query: str, context: List[str],
                                     conversation_memory) -> str:
        """Generate response for statements"""
        # Acknowledge the statement
        acknowledgment = random.choice(self.response_templates['acknowledgment'])
        
        # Check if we have relevant context
        if context:
            # Extract a relevant fact
            cleaned = self._clean_text(context[0])
            try:
                sentences = sent_tokenize(cleaned)[:2]
                if sentences:
                    follow_up = f" This reminds me that {sentences[0].lower()}"
                    return acknowledgment + follow_up
            except:
                pass
        
        return acknowledgment
    
    def _generate_factual_response(self, query: str, context: List[str]) -> str:
        """Generate factual response with better formatting"""
        if not context:
            return random.choice(self.response_templates['uncertain'])

        cleaned = self._clean_text(" ".join(context[:3]))
        
        # Extract a concise answer
        try:
            sentences = sent_tokenize(cleaned)
        except:
            sentences = cleaned.split('. ')
        
        # Try to find sentences that directly answer the query
        query_words = set(self._normalize_query_terms(query))
        relevant_sentences = []
        
        for sent in sentences[:5]:  # Check first 5 sentences
            sent_lower = sent.lower()
            # Check if sentence contains key words from query
            sent_terms = set(self._normalize_query_terms(sent_lower))
            word_overlap = len(query_words.intersection(sent_terms))
            if word_overlap > 1:
                relevant_sentences.append(sent)
        
        if relevant_sentences:
            response = ' '.join(relevant_sentences[:2])
        else:
            response = sentences[0] if sentences else cleaned[:300]
        
        # Remove duplicates if any
        if response.count(response[:50]) > 1:
            response = response[:response.find(response[:50], 50)]
        
        # Add intro
        intro = random.choice(self.response_templates['opinion'])
        return intro + response
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove Wikipedia markup artifacts
        text = re.sub(r'\[\d+\]', '', text)  # Remove citation markers like [1]
        text = re.sub(r'\([^)]*\)', '', text)  # Remove parentheticals that might be metadata
        text = re.sub(r'\s+', ' ', text)
        
        # Remove duplicate sentences (common in Wikipedia extracts)
        sentences = text.split('. ')
        unique_sentences = []
        seen = set()
        for sent in sentences:
            sent_lower = sent.lower()[:50]  # Compare first 50 chars
            if sent_lower not in seen:
                unique_sentences.append(sent)
                seen.add(sent_lower)
        
        return '. '.join(unique_sentences)
    
    def _extract_key_sentences(self, text: str, query: str, max_sentences: int = 3) -> List[str]:
        """Extract key sentences relevant to the query"""
        try:
            sentences = sent_tokenize(text)
        except:
            sentences = text.split('. ')
        
        query_words = set(self._normalize_query_terms(query))
        scored_sentences = []
        
        for idx, sent in enumerate(sentences):
            sent_lower = sent.lower()
            sent_terms = set(self._normalize_query_terms(sent_lower))
            # Score based on word overlap and position
            word_overlap = len(query_words.intersection(sent_terms))
            
            # Boost score if sentence contains definition patterns
            if any(pattern in sent_lower for pattern in ['is a', 'refers to', 'defined as', 'means']):
                word_overlap += 2

            # Penalize question-like lines and quiz prompts often present in datasets
            if sent.strip().endswith('?'):
                word_overlap -= 2
            if any(marker in sent_lower for marker in ['question:', 'answer:', 'quiz', 'choose one']):
                word_overlap -= 2
            
            # Boost score for first sentence (often contains definition)
            if idx == 0:
                word_overlap += 1
            
            # Penalize very short sentences
            if len(sent.split()) < 5:
                word_overlap -= 1
            
            if word_overlap > 0:
                scored_sentences.append((word_overlap, idx, sent))
        
        # Sort by relevance score (higher score) and position (lower idx)
        scored_sentences.sort(key=lambda x: (-x[0], x[1]))
        
        # Return top sentences
        return [sent for score, idx, sent in scored_sentences[:max_sentences]]

    def _normalize_query_terms(self, text: str) -> List[str]:
        """Normalize text into lightweight content-bearing terms."""
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        stop = {
            'the', 'a', 'an', 'of', 'to', 'in', 'on', 'at', 'for', 'and', 'or',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'do', 'does',
            'did', 'what', 'why', 'how', 'when', 'where', 'who', 'which',
            'can', 'could', 'would', 'should', 'tell', 'give', 'show', 'about',
            'me', 'you', 'your', 'my', 'i'
        }
        return [tok for tok in tokens if len(tok) > 2 and tok not in stop]
    
    def _add_personality(self, response: str, conversation_memory) -> str:
        """Add personality to response based on conversation context"""
        try:
            # Adjust based on conversation sentiment if available
            summary = conversation_memory.get_summary()
            avg_sentiment = summary.get('avg_sentiment', 0)
            
            if avg_sentiment > 0.3:
                # Positive conversation - add enthusiasm
                enthusiasm_phrases = [" 😊", " I'm happy to help!", " Great question!", " Wonderful!"]
                if random.random() < 0.3:
                    response += random.choice(enthusiasm_phrases)
            elif avg_sentiment < -0.3:
                # Negative conversation - add empathy
                empathy_phrases = [" I understand your concern.", " That must be difficult.", " I see your point.", " 🤔"]
                if random.random() < 0.3:
                    response += random.choice(empathy_phrases)
        except:
            pass  # Skip personality if memory access fails
        
        return response
    def _humanize_response(self, text: str) -> str:
        """Apply small stylistic tweaks to make a reply sound more human.

        This can include inserting filler words randomly, applying common
        contractions, and occasionally adding a casual sign-off or empathy
        phrase.
        """
        # shorten common phrases using contractions
        for long, short in self.contractions.items():
            text = text.replace(long, short)
        # randomly insert a filler at the start or mid-sentence
        if random.random() < 0.3:
            filler = random.choice(self.filler_words)
            # decide where to insert: beginning or after first comma
            if random.random() < 0.5:
                text = filler.capitalize() + ", " + text
            else:
                parts = text.split(',')
                if len(parts) > 1:
                    parts.insert(1, filler)
                    text = ','.join(parts)
                else:
                    text = filler.capitalize() + ", " + text
        # occasionally tack on a casual phrase
        if random.random() < 0.1:
            text += random.choice([" 🙂", " Hope that helps, Let me know if you need anything else."])
        return text
