#!/usr/bin/env python3
"""
Test script for Shinrai chatbot
"""

from shinrai.core import Shinrai
import sys

def main():
    # Initialize chatbot
    print("Loading Shinrai...")
    shinrai = Shinrai(model_path="shinrai_model")
    
    # Check if we have training data
    if len(shinrai.documents) == 0:
        print("No training data found. Please train first with:")
        print("python shinrai.py train --url <website> --crawl <pages>")
        sys.exit(1)
    
    print(f"Knowledge base: {len(shinrai.documents)} documents")
    print("\n" + "="*60)
    print("Test Queries:")
    print("="*60)
    
    # Test queries
    # verify that the loader handles jsonl files correctly
    import tempfile, json, os
    sample_jsonl = tempfile.NamedTemporaryFile(delete=False, suffix='.jsonl')
    try:
        sample_jsonl.write(b'{"question": "what", "answer": "hello"}\n')
        sample_jsonl.write(b'{"question": "who", "answer": "world"}\n')
        sample_jsonl.flush()
        sample_jsonl.close()
        print(f"\nLoaded texts from sample jsonl: {shinrai._load_from_file(sample_jsonl.name)}")
    finally:
        os.unlink(sample_jsonl.name)

    # now test multi-line object handling
    sample_jsonl = tempfile.NamedTemporaryFile(delete=False, suffix='.jsonl')
    try:
        sample_jsonl.write(b'{\n  "foo": "bar",\n  "baz": 123\n}\n')
        sample_jsonl.flush()
        sample_jsonl.close()
        print(f"\nLoaded texts from multiline jsonl: {shinrai._load_from_file(sample_jsonl.name)}")
    finally:
        os.unlink(sample_jsonl.name)

    test_queries = [
        "what is a language",
        "tell me about english",
        "hello",
        "how are you",
        "goodbye",
        # new edge cases
        "give me some facts about Poland",
        "make a simple python program",
        # provenance check: should reference context or training
        "how do you know that?"
    ]
    
    for query in test_queries:
        print(f"\n👤 You: {query}")
        
        # Get relevant documents
        # for queries that are purely social or code requests there may
        # not be any context; we keep code robust by providing an empty
        # list if no documents are found
        relevant_docs = shinrai._get_relevant_documents(query) or []
        # store last context to allow provenance checks when the query itself is a follow-up
        last_context = relevant_docs

        # Generate response
        response = shinrai.response_generator.generate(
            query,
            relevant_docs,
            shinrai.conversation_memory,
            shinrai.knowledge_graph
        )
        shinrai.conversation_memory.add_interaction(query, response)
        print(f"🤖 Shinrai: {response}")

    # make a follow-up query using the previous response context
    followup = "how do you know that?"
    print(f"\n👤 You: {followup}")
    response = shinrai.response_generator.generate(
        followup,
        last_context,
        shinrai.conversation_memory,
        shinrai.knowledge_graph
    )
    print(f"🤖 Shinrai: {response}")

if __name__ == "__main__":
    main()