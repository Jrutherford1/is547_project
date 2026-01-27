"""
Generate embeddings for documents and store in Neo4j.

This module extracts text from documents in the graph dataset,
generates vector embeddings using Ollama, and stores them as
properties on Document nodes in Neo4j.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

import ollama
from neo4j import GraphDatabase

from data_pipeline.nlp_term_extraction_preview import extract_text

load_dotenv()


def setup_embedder(
    model: str = None,
    host: str = None
) -> Dict[str, str]:
    """
    Initialize Ollama embedder configuration.

    Args:
        model: Embedding model name (default: from OLLAMA_EMBEDDING_MODEL env var)
        host: Ollama host URL (default: from OLLAMA_HOST env var)

    Returns:
        Dict with model and host configuration
    """
    config = {
        "model": model or os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        "host": host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
    }
    print(f"Embedder configured: model={config['model']}, host={config['host']}")
    return config


def get_embedding(text: str, model: str = "nomic-embed-text") -> List[float]:
    """
    Generate embedding for text using Ollama.

    Args:
        text: Text to embed
        model: Ollama embedding model name

    Returns:
        List of floats (embedding vector)
    """
    response = ollama.embed(model=model, input=text)
    return response["embeddings"][0]


def get_neo4j_driver():
    """Get Neo4j driver from environment variables."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")

    if not password:
        raise ValueError("NEO4J_PASSWORD environment variable not set")

    return GraphDatabase.driver(uri, auth=(user, password))


def get_documents_without_embeddings(driver, limit: Optional[int] = None) -> List[Dict]:
    """
    Query Neo4j for documents that don't have embeddings yet.

    Args:
        driver: Neo4j driver
        limit: Optional limit on number of documents

    Returns:
        List of document records with id, name, filePath
    """
    query = """
    MATCH (d:Document)
    WHERE d.embedding IS NULL
    RETURN d.id AS id, d.name AS name, d.filePath AS filePath
    """
    if limit:
        query += f" LIMIT {limit}"

    with driver.session() as session:
        result = session.run(query)
        return [dict(record) for record in result]


def get_all_documents(driver, limit: Optional[int] = None) -> List[Dict]:
    """
    Query Neo4j for all documents.

    Args:
        driver: Neo4j driver
        limit: Optional limit on number of documents

    Returns:
        List of document records with id, name, filePath
    """
    query = """
    MATCH (d:Document)
    RETURN d.id AS id, d.name AS name, d.filePath AS filePath
    """
    if limit:
        query += f" LIMIT {limit}"

    with driver.session() as session:
        result = session.run(query)
        return [dict(record) for record in result]


def store_embedding(driver, doc_id: str, embedding: List[float]):
    """
    Store embedding on a Document node in Neo4j.

    Args:
        driver: Neo4j driver
        doc_id: Document ID
        embedding: Embedding vector as list of floats
    """
    query = """
    MATCH (d:Document {id: $doc_id})
    SET d.embedding = $embedding
    """
    with driver.session() as session:
        session.run(query, doc_id=doc_id, embedding=embedding)


def extract_and_embed_documents(
    driver=None,
    embedder_config: Dict[str, str] = None,
    base_dir: str = "data/committees_processed_for_graph",
    batch_size: int = 50,
    skip_existing: bool = True,
    limit: Optional[int] = None,
    max_text_length: int = 8000
) -> Dict[str, Any]:
    """
    Extract text from documents, generate embeddings, store in Neo4j.

    Args:
        driver: Neo4j driver (creates one if not provided)
        embedder_config: Embedder configuration dict
        base_dir: Base directory containing documents
        batch_size: Number of documents to process before progress report
        skip_existing: Skip documents that already have embeddings
        limit: Optional limit on number of documents to process
        max_text_length: Maximum text length to embed (truncates longer texts)

    Returns:
        Dict with processing statistics
    """
    # Setup
    if driver is None:
        driver = get_neo4j_driver()
        close_driver = True
    else:
        close_driver = False

    if embedder_config is None:
        embedder_config = setup_embedder()

    model = embedder_config["model"]

    print("=" * 60)
    print("GENERATING DOCUMENT EMBEDDINGS")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Base directory: {base_dir}")
    print(f"Skip existing: {skip_existing}")
    print()

    # Get documents to process
    if skip_existing:
        documents = get_documents_without_embeddings(driver, limit)
        print(f"Documents without embeddings: {len(documents)}")
    else:
        documents = get_all_documents(driver, limit)
        print(f"Total documents: {len(documents)}")

    if not documents:
        print("No documents to process.")
        return {"processed": 0, "embedded": 0, "skipped": 0, "errors": 0}

    # Process documents
    stats = {
        "processed": 0,
        "embedded": 0,
        "skipped": 0,
        "errors": 0,
        "error_files": []
    }

    for i, doc in enumerate(documents):
        doc_id = doc["id"]
        file_path = doc.get("filePath")

        if not file_path:
            stats["skipped"] += 1
            continue

        # Build full path
        full_path = Path(base_dir) / file_path
        if not full_path.exists():
            # Try alternative path construction
            full_path = Path(file_path)
            if not full_path.exists():
                stats["skipped"] += 1
                continue

        try:
            # Extract text
            text = extract_text(str(full_path))

            if not text or len(text.strip()) < 100:
                stats["skipped"] += 1
                continue

            # Truncate if needed
            if len(text) > max_text_length:
                text = text[:max_text_length]

            # Generate embedding
            embedding = get_embedding(text, model=model)

            # Store in Neo4j
            store_embedding(driver, doc_id, embedding)

            stats["embedded"] += 1

        except Exception as e:
            stats["errors"] += 1
            stats["error_files"].append((str(full_path), str(e)[:100]))

        stats["processed"] += 1

        # Progress report
        if (i + 1) % batch_size == 0:
            print(f"  Processed {i + 1}/{len(documents)} documents...")

    # Final report
    print()
    print("=" * 60)
    print("EMBEDDING GENERATION COMPLETE")
    print("=" * 60)
    print(f"Documents processed: {stats['processed']}")
    print(f"Embeddings generated: {stats['embedded']}")
    print(f"Skipped (no text/path): {stats['skipped']}")
    print(f"Errors: {stats['errors']}")

    if stats["error_files"]:
        print(f"\nFirst 5 errors:")
        for path, err in stats["error_files"][:5]:
            print(f"  {path}: {err}")

    if close_driver:
        driver.close()

    return stats


def check_embedding_status(driver=None) -> Dict[str, int]:
    """
    Check how many documents have embeddings.

    Args:
        driver: Neo4j driver (creates one if not provided)

    Returns:
        Dict with counts: total, with_embeddings, without_embeddings
    """
    if driver is None:
        driver = get_neo4j_driver()
        close_driver = True
    else:
        close_driver = False

    query = """
    MATCH (d:Document)
    RETURN
        count(d) AS total,
        count(d.embedding) AS with_embeddings
    """

    with driver.session() as session:
        result = session.run(query).single()
        stats = {
            "total": result["total"],
            "with_embeddings": result["with_embeddings"],
            "without_embeddings": result["total"] - result["with_embeddings"]
        }

    print(f"Embedding Status:")
    print(f"  Total documents: {stats['total']}")
    print(f"  With embeddings: {stats['with_embeddings']}")
    print(f"  Without embeddings: {stats['without_embeddings']}")

    if close_driver:
        driver.close()

    return stats


if __name__ == "__main__":
    # Test embedding generation
    config = setup_embedder()
    stats = extract_and_embed_documents(limit=5)
    check_embedding_status()
