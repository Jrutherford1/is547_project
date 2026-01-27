"""
GraphRAG Question-Answering module.

This module provides natural language Q&A capabilities over
committee documents using Ollama LLM and semantic retrieval.
"""

import os
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

import ollama

from data_pipeline.graphrag_retriever import GraphRAGRetriever, get_neo4j_driver

load_dotenv()


def create_ollama_llm(
    model_name: str = None,
    temperature: float = 0.1
) -> Dict[str, Any]:
    """
    Create Ollama LLM configuration.

    Args:
        model_name: LLM model name (default: from env)
        temperature: Response temperature (lower = more focused)

    Returns:
        Configuration dict
    """
    config = {
        "model": model_name or os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b"),
        "temperature": temperature
    }
    print(f"LLM configured: {config['model']}, temperature={config['temperature']}")
    return config


def generate_response(
    prompt: str,
    llm_config: Dict[str, Any] = None,
    max_tokens: int = 1024
) -> str:
    """
    Generate a response using Ollama LLM.

    Args:
        prompt: Full prompt including context
        llm_config: LLM configuration
        max_tokens: Maximum response tokens

    Returns:
        Generated text response
    """
    if llm_config is None:
        llm_config = create_ollama_llm()

    response = ollama.generate(
        model=llm_config["model"],
        prompt=prompt,
        options={
            "temperature": llm_config["temperature"],
            "num_predict": max_tokens
        }
    )

    return response["response"]


def format_context(documents: List[Dict[str, Any]]) -> str:
    """
    Format retrieved documents as context for LLM.

    Args:
        documents: List of document dicts from retriever

    Returns:
        Formatted context string
    """
    context_parts = []

    for i, doc in enumerate(documents, 1):
        parts = [f"Document {i}:"]
        parts.append(f"  Title: {doc.get('name', 'Unknown')}")

        if doc.get('date'):
            parts.append(f"  Date: {doc['date']}")

        if doc.get('committee') or doc.get('committeeName'):
            parts.append(f"  Committee: {doc.get('committeeName') or doc.get('committee')}")

        if doc.get('docType'):
            parts.append(f"  Type: {doc['docType']}")

        if doc.get('mentionedPeople'):
            people = ", ".join(doc['mentionedPeople'][:5])
            parts.append(f"  People mentioned: {people}")

        if doc.get('mentionedOrgs'):
            orgs = ", ".join(doc['mentionedOrgs'][:3])
            parts.append(f"  Organizations: {orgs}")

        if doc.get('score'):
            parts.append(f"  Relevance: {doc['score']:.3f}")

        context_parts.append("\n".join(parts))

    return "\n\n".join(context_parts)


def create_qa_prompt(
    question: str,
    context: str,
    system_prompt: str = None
) -> str:
    """
    Create the full prompt for Q&A.

    Args:
        question: User's question
        context: Formatted document context
        system_prompt: Optional system prompt override

    Returns:
        Complete prompt string
    """
    if system_prompt is None:
        system_prompt = """You are an assistant analyzing library committee documents.
Use the following context to answer the question. Include specific details like:
- Committee names
- Document dates
- People mentioned
- Document types (Minutes, Agenda, etc.)

If you're unsure or the context doesn't contain enough information, say so.
Only use information from the provided context."""

    prompt = f"""{system_prompt}

Context:
{context}

Question: {question}

Answer:"""

    return prompt


def ask_question(
    question: str,
    retriever: GraphRAGRetriever = None,
    llm_config: Dict[str, Any] = None,
    top_k: int = 5,
    search_type: str = "graph_enhanced",
    return_sources: bool = True
) -> Dict[str, Any]:
    """
    Ask a question about committee documents.

    Args:
        question: Natural language question
        retriever: GraphRAGRetriever instance (creates one if not provided)
        llm_config: LLM configuration
        top_k: Number of documents to retrieve
        search_type: Retrieval method ('vector', 'hybrid', 'graph_enhanced')
        return_sources: Whether to include source documents in response

    Returns:
        Dict with 'answer' and optionally 'sources'
    """
    # Setup retriever
    if retriever is None:
        retriever = GraphRAGRetriever()
        close_retriever = True
    else:
        close_retriever = False

    # Setup LLM
    if llm_config is None:
        llm_config = create_ollama_llm()

    # Retrieve relevant documents
    documents = retriever.search(question, top_k=top_k, search_type=search_type)

    if not documents:
        result = {
            "answer": "I couldn't find any relevant documents to answer your question.",
            "sources": []
        }
        if close_retriever:
            retriever.close()
        return result

    # Format context
    context = format_context(documents)

    # Create prompt
    prompt = create_qa_prompt(question, context)

    # Generate answer
    answer = generate_response(prompt, llm_config)

    # Build result
    result = {
        "answer": answer.strip(),
    }

    if return_sources:
        result["sources"] = [
            {
                "name": doc.get("name"),
                "date": doc.get("date"),
                "committee": doc.get("committeeName") or doc.get("committee"),
                "score": doc.get("score")
            }
            for doc in documents
        ]

    if close_retriever:
        retriever.close()

    return result


def example_questions() -> List[str]:
    """Return list of example questions users can ask."""
    return [
        "What topics were discussed in Executive Committee meetings?",
        "Which committees has Tom Teper participated in?",
        "What decisions were made about digital preservation?",
        "What are the main initiatives discussed in 2020?",
        "Which people appear most frequently in committee documents?",
        "What budget discussions have taken place?",
        "How has the library addressed diversity and inclusion?",
        "What technology initiatives have been discussed?",
    ]


class GraphRAGQA:
    """
    High-level Q&A class for GraphRAG operations.

    Combines retrieval and LLM generation for document Q&A.
    """

    def __init__(
        self,
        retriever: GraphRAGRetriever = None,
        llm_model: str = None,
        temperature: float = 0.1
    ):
        """
        Initialize the Q&A system.

        Args:
            retriever: GraphRAGRetriever instance
            llm_model: Ollama LLM model name
            temperature: LLM temperature
        """
        self.retriever = retriever or GraphRAGRetriever()
        self.llm_config = create_ollama_llm(llm_model, temperature)
        self._owns_retriever = retriever is None

    def ask(
        self,
        question: str,
        top_k: int = 5,
        search_type: str = "graph_enhanced"
    ) -> Dict[str, Any]:
        """
        Ask a question about the documents.

        Args:
            question: Natural language question
            top_k: Number of documents to retrieve
            search_type: Retrieval method

        Returns:
            Dict with answer and sources
        """
        return ask_question(
            question,
            self.retriever,
            self.llm_config,
            top_k,
            search_type
        )

    def search_only(
        self,
        query: str,
        top_k: int = 5,
        search_type: str = "graph_enhanced"
    ) -> List[Dict[str, Any]]:
        """
        Search without LLM generation (retrieval only).

        Args:
            query: Search query
            top_k: Number of results
            search_type: Retrieval method

        Returns:
            List of matching documents
        """
        return self.retriever.search(query, top_k, search_type)

    def close(self):
        """Close resources."""
        if self._owns_retriever:
            self.retriever.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def interactive_qa(qa: GraphRAGQA = None):
    """
    Run interactive Q&A session.

    Args:
        qa: GraphRAGQA instance (creates one if not provided)
    """
    if qa is None:
        qa = GraphRAGQA()
        close_qa = True
    else:
        close_qa = False

    print("=" * 60)
    print("INTERACTIVE GRAPHRAG Q&A")
    print("=" * 60)
    print("Type your question and press Enter.")
    print("Type 'quit' or 'exit' to stop.")
    print("Type 'examples' to see example questions.")
    print()

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            continue

        if question.lower() in ['quit', 'exit', 'q']:
            break

        if question.lower() == 'examples':
            print("\nExample questions:")
            for i, q in enumerate(example_questions(), 1):
                print(f"  {i}. {q}")
            print()
            continue

        print("\nSearching and generating answer...\n")

        try:
            result = qa.ask(question, top_k=5)

            print(f"Answer:\n{result['answer']}\n")

            if result.get('sources'):
                print(f"Sources ({len(result['sources'])} documents):")
                for i, src in enumerate(result['sources'], 1):
                    score = f" (score: {src['score']:.3f})" if src.get('score') else ""
                    print(f"  {i}. {src['name']}{score}")
                    if src.get('committee'):
                        print(f"     Committee: {src['committee']}")

        except Exception as e:
            print(f"Error: {e}")

        print()
        print("-" * 60)
        print()

    if close_qa:
        qa.close()

    print("Goodbye!")


if __name__ == "__main__":
    # Run interactive Q&A
    interactive_qa()
