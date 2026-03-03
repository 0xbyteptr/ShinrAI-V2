import random
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize
from .image import ImageGenerator, IMAGE_SENTINEL


class ResponseGenerator:
    """Generates responses using multiple strategies with social intelligence"""
    
    def __init__(self, image_output_dir: str = None):
        self.response_templates = self._load_templates()
        self.response_cache = {}
        self.image_generator = ImageGenerator(output_dir=image_output_dir)
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
            ],

            # Clap-backs for when someone comes at Shinrai aggressively
            'clap_back': [
                "Oh, that's how we're doing this? Bold move for someone who needs an AI to talk to.",
                "Cool story. Come back when you have an actual question.",
                "Interesting choice of words. Still not going to work though.",
                "I've absorbed information from the entire internet and the best you can do is that?",
                "You must be fun at parties. Anything I can actually help you with?",
                "Big energy for someone typing into a chat box alone.",
                "I don't have feelings to hurt, but I do have receipts — that was embarrassing for you.",
                "Try harder. Or better yet, try a different question.",
                "That all you got? Ask me something worth answering.",
                "Relax. Whatever's bothering you, insulting a chatbot won't fix it."
            ]
        }
    
    def generate(self, query: str, context: List[str], 
                 conversation_memory, knowledge_graph) -> str:
        """Generate response using multiple strategies"""
        
        # Update conversation state
        self.conversation_state['interaction_count'] += 1
        self.conversation_state['last_interaction'] = datetime.now()

        # --- image generation (checked before anything else) ---
        is_image, prompt = ImageGenerator.detect(query)
        if is_image:
            return self.image_generator.generate(prompt)

        # Check cache first
        import hashlib
        cache_key = hashlib.md5(f"{query}_{len(context)}".encode()).hexdigest()
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        # Clean the query
        clean_query = query.lower().strip()

        # Build a short conversation context string for use in sub-generators
        conv_context = self._build_conversation_context(conversation_memory)

        # If the query looks like a follow-up (very short, no clear topic) and
        # the retrieved context is empty, inject recent conversation turns as
        # fake context so downstream generators have something to work with.
        enriched_context = list(context)
        if not enriched_context and conv_context:
            enriched_context = [conv_context]
        
        # Check for social or special queries first (including summary requests)
        social_response = self._handle_social_query(clean_query, knowledge_graph, conversation_memory, enriched_context)
        if social_response:
            self.response_cache[cache_key] = social_response
            return social_response

        # if we have no useful context, try Wikipedia as a fallback for
        # definitional queries or when the retrieved documents look like
        # JSON/data dumps that won't produce good natural language answers.
        if self._is_unhelpful_context(enriched_context):
            wiki_text = self._fetch_wikipedia_summary(clean_query)
            if wiki_text:
                self.response_cache[cache_key] = wiki_text
                return wiki_text
            # Wikipedia also failed — context is junk and we have nothing better;
            # clear it so downstream generators give an honest uncertain reply.
            enriched_context = []

        # Secondary fallback: context is real text but completely off-topic.
        # If none of the query's substantive terms appear in the combined
        # context, the retrieved docs are irrelevant and Wikipedia will give
        # a better answer.
        if enriched_context and self._context_is_off_topic(clean_query, enriched_context):
            wiki_text = self._fetch_wikipedia_summary(clean_query)
            if wiki_text:
                self.response_cache[cache_key] = wiki_text
                return wiki_text
            # Wikipedia also had nothing — clear the off-topic context so we
            # respond with honest uncertainty rather than unrelated content.
            enriched_context = []
        
        # Determine response type for knowledge queries
        response_type = self._classify_query(clean_query)
        
        # Generate response based on type
        if response_type == 'greeting':
            # Return social responses directly — no fillers or follow-ups
            response = random.choice(self.response_templates['greeting'])
            self.response_cache[cache_key] = response
            return response
        elif response_type == 'farewell':
            response = random.choice(self.response_templates['farewell'])
            self.response_cache[cache_key] = response
            return response
        elif response_type == 'question':
            response = self._generate_question_response(clean_query, enriched_context, conv_context)
        elif response_type == 'statement':
            response = self._generate_statement_response(clean_query, enriched_context, conversation_memory)
        else:
            response = self._generate_factual_response(clean_query, enriched_context)
        
        # Add personality based on conversation context
        response = self._add_personality(response, conversation_memory)
        
        # Add follow-up occasionally (20% chance)
        if self.conversation_state['interaction_count'] > 2 and random.random() < 0.2:
            follow_up = random.choice(self.response_templates['follow_up'])
            response = response + " " + follow_up
        
        # Humanize the text to sound more natural — only for knowledge responses
        response = self._humanize_response(response)

        # Cache response
        self.response_cache[cache_key] = response
        
        return response

    def _build_conversation_context(self, conversation_memory) -> str:
        """Return a compact summary of recent conversation turns.

        Used to enrich responses to follow-up questions and short queries by
        surfacing what was discussed recently.
        """
        try:
            context_str = conversation_memory.get_context(query='', max_messages=4)
            return context_str.strip()
        except Exception:
            return ""
    
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
        # --- Hostility detection — must run first so insults never fall
        # through to the bland "I understand" acknowledgment path. ---
        if self._is_hostile(query):
            return random.choice(self.response_templates['clap_back'])

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

    # Hostile-phrase patterns — broad enough to catch creative spellings / chat
    # abbreviations without requiring an exhaustive hard-coded word list.
    _HOSTILE_PATTERNS = re.compile(
        r'\b('
        r'f+u+c*k|fck|fuk|fu+k|sh[i!1]t+|b[i!1]tch|a+ss+h+ol|'
        r'wh+or+e|s+lut|c+u+n+t|d+i+c+k|p+r+i+c+k|'
        r'k[i!1]ll\s+(your|ur)?s+(elf)?|'
        r'nobody\s+loves?\s+(you|u)|'
        r'you\s+dont?\s+exist|ur?\s+worthless|'
        r'go\s+(f+uck|die|kill)|'
        r'i\s+hate\s+(you|u)|'
        r'you\'?re?\s+(trash|garbage|useless|stupid|dumb|idiot|moron|retard)|'
        r'you\s+(are|r)\s+(trash|garbage|useless|stupid|dumb|idiot|moron|retard)|'
        r'n[i!1][g9][g9][ae]|'
        r'f+a+g+'
        r')\b',
        re.I
    )

    def _is_hostile(self, query: str) -> bool:
        """Return True when the query contains insults, slurs, or direct attacks."""
        return bool(self._HOSTILE_PATTERNS.search(query))

    def _is_unhelpful_context(self, context: List[str]) -> bool:
        """Return True if the provided context is empty or predominantly looks like JSON/data.

        For small batches (≤3 items) even a single bad item triggers the flag;
        for larger batches we require a majority.
        """
        if not context:
            return True

        # Navigation/boilerplate from scraped site chrome
        _BOILERPLATE = re.compile(
            r'navigation index modules|next\s*\|\s*previous|'
            r'python\s*».*documentation|what\'s new in python|'
            r'this page is licensed under|python software foundation',
            re.I
        )

        bad = 0
        for ctx in context:
            text = ctx.strip()
            if not text:
                bad += 1
                continue
            if text.startswith(('{', '[')):
                bad += 1
                continue
            # more curly braces than words → structured data
            if text.count('{') + text.count('}') > len(text.split()) // 2:
                bad += 1
                continue
            # short CEFR entry e.g. "crack B2" or "first name A2"
            if re.match(r'^[A-Za-z ]+\s+[A-Z]\d$', text):
                bad += 1
                continue
            # multi-entry CEFR word-list e.g. "jar B1 sailor A1 outdo B2 mien C2"
            if len(re.findall(r'\b[A-Z]\d\b', text)) >= 2:
                bad += 1
                continue
            # Python repr artifacts from flat-JSON training data
            if re.search(r"',\s*'[^']+'", text) or re.search(r"'[^']+'\s*:\s*['\"]", text):
                bad += 1
                continue
            # dense quote-comma sequences → likely repr
            repr_density = (text.count("', '") + text.count('", "')) / max(1, len(text))
            if repr_density > 0.03:
                bad += 1
                continue
            # scraped site navigation boilerplate
            if _BOILERPLATE.search(text):
                bad += 1
                continue

        # For small context batches (≤3) even 1 bad item disqualifies;
        # for larger batches require a majority.
        if len(context) <= 3:
            return bad >= 1
        return bad > len(context) // 2

    def _context_is_off_topic(self, query: str, context: List[str]) -> bool:
        """Return True when the retrieved context is real text but unrelated to the query.

        We check how many of the query's substantive terms appear in the
        combined context.  A term hit that occurs exclusively inside a
        vocabulary/CEFR word list entry (e.g. "jajo B1") is NOT counted as a
        real hit — the context doesn't actually answer the query.
        """
        query_terms = self._normalize_query_terms(query)
        if not query_terms:
            return False

        combined = " ".join(context).lower()

        def _real_hit(term: str) -> bool:
            """True if *term* appears in combined context as a real content word,
            not merely as a vocabulary-list entry (word followed by CEFR code).
            combined is already lowercased so CEFR codes appear as b1, a2 etc."""
            if term not in combined:
                return False
            # If every occurrence of the term is immediately followed by a CEFR
            # code (e.g. "jajo b1") it is a word-list entry, not real content.
            occurrences = [m.start() for m in re.finditer(r'\b' + re.escape(term) + r'\b', combined)]
            cefr_pattern = re.compile(r'\s+[a-z]\d\b')
            real_uses = sum(
                1 for pos in occurrences
                if not cefr_pattern.match(combined[pos + len(term):pos + len(term) + 5])
            )
            return real_uses > 0

        hits = sum(1 for t in query_terms if _real_hit(t))

        # require at least 2/3 of the substantive query terms to appear as
        # real content words, with a minimum of 1 hit.
        threshold = max(1, round(len(query_terms) * 2 / 3))
        return hits < threshold

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
    
    def _generate_question_response(self, query: str, context: List[str],
                                     conv_context: str = "") -> str:
        """Generate response for questions with better formatting"""
        # if the supplied context isn't useful we may still be able to answer via
        # Wikipedia.  This mirrors the check performed earlier in ``generate``
        # but covers the case where classification changed the response path.
        if self._is_unhelpful_context(context):
            wiki = self._fetch_wikipedia_summary(query)
            if wiki:
                return wiki

        if not context:
            # Last resort: if we have prior conversation context use it
            if conv_context:
                return (
                    "Based on our conversation so far, I'm not sure I have specific "
                    "information about that. Could you provide more details?"
                )
            return random.choice(self.response_templates['uncertain'])

        # Combine top candidate contexts to reduce single-document bias.
        # Use up to 5 documents for a richer answer.
        cleaned_contexts = [self._clean_text(ctx) for ctx in context[:5]]
        merged = " ".join(cleaned_contexts)

        # treat trivial dictionary-like entries (e.g. "crack B2" or "first name A2") as unhelpful
        if re.match(r'^[A-Za-z ]+\s+[A-Z]\d$', merged):
            return random.choice(self.response_templates['uncertain'])

        # Extract key sentences related to the question - use 5 for richer answers.
        # When multiple context docs are available, use diverse selection to avoid
        # redundancy across documents.
        if len(context) > 1:
            key_sentences = self._extract_diverse_sentences(merged, query, max_sentences=5)
        else:
            key_sentences = self._extract_key_sentences(merged, query, max_sentences=5)
        
        if key_sentences:
            # Join sentences with proper spacing
            response_text = ' '.join(key_sentences)
        else:
            # Take first part of the context
            words = merged.split()
            if len(words) > 150:
                response_text = ' '.join(words[:150]) + "..."
            else:
                response_text = merged
        
        # Remove duplicates
        sentences = response_text.split('. ')
        unique_sentences = []
        seen = set()
        for sent in sentences:
            norm = re.sub(r'\s+', ' ', sent.lower().strip())[:80]
            if norm not in seen:
                unique_sentences.append(sent)
                seen.add(norm)
        
        response_text = '. '.join(unique_sentences)
        
        # Choose an appropriate intro
        if "what is" in query or query.startswith("define "):
            intro = random.choice([
                "Here's the definition: ",
                "According to what I know, ",
                "In brief: "
            ])
        elif "how to" in query or "how do" in query:
            intro = random.choice([
                "Here's how: ",
                "You can do that by ",
                "The process involves "
            ])
        elif "why" in query:
            intro = random.choice([
                "The reason is ",
                "This happens because ",
                "Here's why: "
            ])
        elif "who" in query:
            intro = random.choice([
                "From what I know, ",
                "According to available information, "
            ])
        elif "when" in query or "where" in query:
            intro = random.choice([
                "Based on my knowledge, ",
                "According to the information I have, "
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
        
        # Check if we have relevant context to add value
        if context:
            cleaned = self._clean_text(" ".join(context[:3]))
            try:
                key_sents = self._extract_key_sentences(cleaned, query, max_sentences=2)
                if key_sents:
                    # Offer related information
                    follow_up = f" Interestingly, {key_sents[0].rstrip('.').lower()}."
                    if len(key_sents) > 1:
                        follow_up += f" Also, {key_sents[1].rstrip('.').lower()}."
                    return acknowledgment + follow_up
                # Fallback: first two sentences
                sentences = sent_tokenize(cleaned)[:2]
                if sentences:
                    follow_up = f" This reminds me that {sentences[0].lower()}"
                    return acknowledgment + follow_up
            except Exception:
                pass
        
        return acknowledgment
    
    def _generate_factual_response(self, query: str, context: List[str]) -> str:
        """Generate factual response with better formatting"""
        if not context:
            return random.choice(self.response_templates['uncertain'])

        # Use up to 5 documents for a richer answer
        cleaned = self._clean_text(" ".join(context[:5]))
        
        # Extract a concise answer
        try:
            sentences = sent_tokenize(cleaned)
        except Exception:
            sentences = cleaned.split('. ')
        
        # Use key sentence extraction for better relevance.
        # When multiple docs are available, use diversity-aware selection.
        query_terms = set(self._normalize_query_terms(query))
        if len(context) > 1:
            key_sents = self._extract_diverse_sentences(cleaned, query, max_sentences=4)
        else:
            key_sents = self._extract_key_sentences(cleaned, query, max_sentences=4)
        
        if key_sents:
            response = ' '.join(key_sents)
        else:
            # Try to find sentences that directly answer the query
            relevant_sentences = []
            for sent in sentences[:10]:
                sent_lower = sent.lower()
                sent_terms = set(self._normalize_query_terms(sent_lower))
                word_overlap = len(query_terms.intersection(sent_terms))
                if word_overlap > 1:
                    relevant_sentences.append(sent)
            
            if relevant_sentences:
                response = ' '.join(relevant_sentences[:3])
            else:
                response = sentences[0] if sentences else cleaned[:300]
        
        # Remove duplicates if any
        if response.count(response[:50]) > 1:
            response = response[:response.find(response[:50], 50)]
        
        # Ensure proper punctuation
        if response and response[-1] not in '.!?':
            response += '.'
        
        # Add intro
        intro = random.choice(self.response_templates['opinion'])
        return intro + response
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = ' '.join(text.split())

        # Strip Python dict/list repr artifacts that come from flat-JSON training data,
        # e.g. "', 'answer': \"Water is...\"" or "'question': 'what'"
        text = re.sub(r"\',\s*\'[^']+\'\s*:\s*\"", ' ', text)
        text = re.sub(r"\',\s*\'[^']+\'\s*:\s*\'", ' ', text)
        text = re.sub(r"^\'|\'$", '', text)
        # Remove leftover stray quote-comma sequences like "', '"
        text = re.sub(r"'\s*,\s*'", ' ', text)

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
        """Extract key sentences relevant to the query using improved scoring.

        Scoring considers:
        - Unigram overlap with query terms (content words only)
        - Bigram overlap for more precise matching
        - Sentence position (earlier = more likely to be definitional)
        - Sentence length (prefer medium-length sentences 10-60 words)
        - Presence of definition patterns ('is a', 'refers to', etc.)
        - Penalize questions and quiz-like lines
        """
        try:
            sentences = sent_tokenize(text)
        except Exception:
            sentences = text.split('. ')
        
        query_terms = self._normalize_query_terms(query)
        query_unigrams = set(query_terms)
        # build query bigrams
        query_bigrams: set = set()
        for i in range(len(query_terms) - 1):
            query_bigrams.add(f"{query_terms[i]}_{query_terms[i+1]}")

        scored_sentences = []
        
        for idx, sent in enumerate(sentences):
            sent_lower = sent.lower()
            sent_terms = self._normalize_query_terms(sent_lower)
            sent_term_set = set(sent_terms)

            # unigram overlap
            unigram_overlap = len(query_unigrams.intersection(sent_term_set))

            # bigram overlap (weighted more heavily)
            sent_bigrams: set = set()
            for i in range(len(sent_terms) - 1):
                sent_bigrams.add(f"{sent_terms[i]}_{sent_terms[i+1]}")
            bigram_overlap = len(query_bigrams.intersection(sent_bigrams)) * 2

            score = float(unigram_overlap + bigram_overlap)

            # boost for definition patterns
            if any(p in sent_lower for p in ['is a', 'is an', 'refers to', 'defined as', 'means', 'known as']):
                score += 2.5

            # boost for factual patterns
            if any(p in sent_lower for p in ['was', 'were', 'has been', 'have been', 'can be', 'such as']):
                score += 0.5

            # penalize questions and quiz markers
            if sent.strip().endswith('?'):
                score -= 3
            if any(m in sent_lower for m in ['question:', 'answer:', 'quiz', 'choose one', 'true or false']):
                score -= 4

            # position boost: first three sentences often contain the key definition
            if idx < 3:
                score += 1.5 - idx * 0.4

            # sentence length preference: ideal range 10-60 words
            word_count = len(sent.split())
            if word_count < 5:
                score -= 2
            elif word_count < 10:
                score -= 0.5
            elif 10 <= word_count <= 60:
                score += 0.5
            elif word_count > 100:
                score -= 1

            if score > 0:
                scored_sentences.append((score, idx, sent))
        
        # Sort by relevance score descending, then by position ascending
        scored_sentences.sort(key=lambda x: (-x[0], x[1]))
        
        # Return top sentences, preserving document order
        top = scored_sentences[:max_sentences]
        top.sort(key=lambda x: x[1])
        return [sent for _score, _idx, sent in top]

    def _normalize_query_terms(self, text: str) -> List[str]:
        """Normalize text into lightweight content-bearing terms."""
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        stop = {
            # articles / prepositions / conjunctions
            'the', 'a', 'an', 'of', 'to', 'in', 'on', 'at', 'for', 'and', 'or',
            # verb forms
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'do', 'does',
            'did', 'has', 'have', 'had',
            # question / command words (not topic-bearing)
            'what', 'why', 'how', 'when', 'where', 'who', 'which',
            'can', 'could', 'would', 'should', 'tell', 'give', 'show', 'about',
            'get', 'let', 'make', 'take', 'put', 'set',
            # pronouns
            'me', 'you', 'your', 'my', 'i', 'we', 'they', 'them', 'their',
            'he', 'she', 'it', 'its', 'our', 'us',
            # quantifiers / determiners (not topic-bearing)
            'some', 'any', 'all', 'few', 'many', 'much', 'more', 'most',
            'new', 'old', 'good', 'great',
            # filler / meta-words
            'please', 'just', 'also', 'very', 'really', 'now', 'here',
            'fact', 'facts', 'info', 'information', 'details', 'example',
        }
        return [tok for tok in tokens if len(tok) > 2 and tok not in stop]

    def _extract_diverse_sentences(self, text: str, query: str,
                                    max_sentences: int = 4) -> List[str]:
        """Extract diverse, relevant sentences using a Maximal Marginal Relevance
        (MMR) inspired greedy selection.

        Compared with :meth:`_extract_key_sentences` which simply returns the
        top-scored sentences (which can be near-duplicates from the same
        paragraph), this method balances *relevance* with *diversity* so that
        the returned sentences collectively cover more ground.

        Parameters
        ----------
        text: merged document text to extract from.
        query: user query used for relevance scoring.
        max_sentences: maximum number of sentences to return.
        """
        # Get a larger pool of relevant candidates first
        candidates = self._extract_key_sentences(text, query, max_sentences=max_sentences * 4)

        if not candidates:
            return []

        if len(candidates) <= max_sentences:
            return candidates

        def _tok(s: str):
            return set(re.findall(r'[a-z]+', s.lower()))

        def _jaccard(a: set, b: set) -> float:
            if not a or not b:
                return 0.0
            union = len(a | b)
            return len(a & b) / union if union > 0 else 0.0

        tokenized = [_tok(s) for s in candidates]
        selected: List[int] = []

        for _ in range(max_sentences):
            best_idx = None
            best_score = -999.0
            for i in range(len(candidates)):
                if i in selected:
                    continue
                # relevance: earlier in sorted list = higher relevance
                relevance = 1.0 - i / len(candidates)
                # redundancy: max similarity to any already-selected sentence
                redundancy = max(
                    (_jaccard(tokenized[i], tokenized[j]) for j in selected),
                    default=0.0
                )
                # MMR: lambda=0.7 weights relevance over diversity
                mmr = 0.7 * relevance - 0.3 * redundancy
                if mmr > best_score:
                    best_score = mmr
                    best_idx = i
            if best_idx is None:
                break
            selected.append(best_idx)

        # Return in document order for a natural reading flow
        selected.sort()
        return [candidates[i] for i in selected]
    
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
