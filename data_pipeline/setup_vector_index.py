"""
Setup Neo4j vector and fulltext indexes for GraphRAG.

This module creates the necessary indexes for semantic search:
- Vector index on Document.embedding for similarity search
- Fulltext index on Document text properties for keyword matching
"""

import os
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from neo4j import GraphDatabase

load_dotenv()


def get_neo4j_driver():
    """Get Neo4j driver from environment variables."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")

    if not password:
        raise ValueError("NEO4J_PASSWORD environment variable not set")

    return GraphDatabase.driver(uri, auth=(user, password))


def create_document_vector_index(
    driver=None,
    index_name: str = "document_embeddings",
    dimensions: int = None,
    similarity_function: str = "cosine"
) -> bool:
    """
    Create vector index on Document.embedding property.

    Args:
        driver: Neo4j driver (creates one if not provided)
        index_name: Name for the vector index
        dimensions: Embedding dimensions (default: from EMBEDDING_DIMENSIONS env var)
        similarity_function: Similarity function ('cosine' or 'euclidean')

    Returns:
        True if index created or already exists
    """
    if driver is None:
        driver = get_neo4j_driver()
        close_driver = True
    else:
        close_driver = False

    if dimensions is None:
        dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", 768))

    print(f"Creating vector index '{index_name}'...")
    print(f"  Dimensions: {dimensions}")
    print(f"  Similarity: {similarity_function}")

    # Check if index already exists
    with driver.session() as session:
        result = session.run("SHOW INDEXES")
        existing = [r["name"] for r in result]

        if index_name in existing:
            print(f"  Index '{index_name}' already exists.")
            if close_driver:
                driver.close()
            return True

    # Create the vector index
    query = f"""
    CREATE VECTOR INDEX `{index_name}`
    FOR (d:Document) ON (d.embedding)
    OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: {dimensions},
            `vector.similarity_function`: '{similarity_function}'
        }}
    }}
    """

    try:
        with driver.session() as session:
            session.run(query)
        print(f"  Vector index '{index_name}' created.")
        success = True
    except Exception as e:
        print(f"  Error creating vector index: {e}")
        success = False

    if close_driver:
        driver.close()

    return success


def create_document_fulltext_index(
    driver=None,
    index_name: str = "document_fulltext"
) -> bool:
    """
    Create fulltext index on Document text properties.

    Args:
        driver: Neo4j driver (creates one if not provided)
        index_name: Name for the fulltext index

    Returns:
        True if index created or already exists
    """
    if driver is None:
        driver = get_neo4j_driver()
        close_driver = True
    else:
        close_driver = False

    print(f"Creating fulltext index '{index_name}'...")

    # Check if index already exists
    with driver.session() as session:
        result = session.run("SHOW INDEXES")
        existing = [r["name"] for r in result]

        if index_name in existing:
            print(f"  Index '{index_name}' already exists.")
            if close_driver:
                driver.close()
            return True

    # Create fulltext index on name and description
    query = f"""
    CREATE FULLTEXT INDEX `{index_name}`
    FOR (d:Document)
    ON EACH [d.name, d.description, d.additionalType]
    """

    try:
        with driver.session() as session:
            session.run(query)
        print(f"  Fulltext index '{index_name}' created.")
        success = True
    except Exception as e:
        print(f"  Error creating fulltext index: {e}")
        success = False

    if close_driver:
        driver.close()

    return success


def wait_for_index_online(
    driver=None,
    index_name: str = "document_embeddings",
    timeout: int = 300,
    poll_interval: int = 5
) -> bool:
    """
    Wait for an index to reach ONLINE state.

    Args:
        driver: Neo4j driver (creates one if not provided)
        index_name: Name of the index to wait for
        timeout: Maximum seconds to wait
        poll_interval: Seconds between status checks

    Returns:
        True if index is online, False if timeout
    """
    if driver is None:
        driver = get_neo4j_driver()
        close_driver = True
    else:
        close_driver = False

    print(f"Waiting for index '{index_name}' to come online...")

    start_time = time.time()

    while time.time() - start_time < timeout:
        with driver.session() as session:
            result = session.run("SHOW INDEXES")
            for record in result:
                if record["name"] == index_name:
                    state = record["state"]
                    if state == "ONLINE":
                        print(f"  Index '{index_name}' is ONLINE.")
                        if close_driver:
                            driver.close()
                        return True
                    else:
                        print(f"  Index state: {state}...")

        time.sleep(poll_interval)

    print(f"  Timeout waiting for index '{index_name}'.")
    if close_driver:
        driver.close()
    return False


def verify_indexes(driver=None) -> List[Dict[str, Any]]:
    """
    List all indexes and their status.

    Args:
        driver: Neo4j driver (creates one if not provided)

    Returns:
        List of index information dicts
    """
    if driver is None:
        driver = get_neo4j_driver()
        close_driver = True
    else:
        close_driver = False

    with driver.session() as session:
        result = session.run("SHOW INDEXES")
        indexes = []
        for record in result:
            indexes.append({
                "name": record["name"],
                "type": record["type"],
                "state": record["state"],
                "labelsOrTypes": record.get("labelsOrTypes", []),
                "properties": record.get("properties", [])
            })

    if close_driver:
        driver.close()

    return indexes


def setup_all_indexes(
    driver=None,
    vector_index_name: str = "document_embeddings",
    fulltext_index_name: str = "document_fulltext",
    dimensions: int = None,
    wait_for_online: bool = True
) -> Dict[str, bool]:
    """
    Create all indexes needed for GraphRAG.

    Args:
        driver: Neo4j driver (creates one if not provided)
        vector_index_name: Name for vector index
        fulltext_index_name: Name for fulltext index
        dimensions: Embedding dimensions
        wait_for_online: Whether to wait for indexes to come online

    Returns:
        Dict with success status for each index
    """
    if driver is None:
        driver = get_neo4j_driver()
        close_driver = True
    else:
        close_driver = False

    print("=" * 60)
    print("SETTING UP GRAPHRAG INDEXES")
    print("=" * 60)
    print()

    results = {}

    # Create fulltext index first (faster)
    results["fulltext"] = create_document_fulltext_index(driver, fulltext_index_name)

    # Create vector index
    results["vector"] = create_document_vector_index(driver, vector_index_name, dimensions)

    # Wait for indexes to come online
    if wait_for_online:
        print()
        if results["fulltext"]:
            wait_for_index_online(driver, fulltext_index_name, timeout=60)
        if results["vector"]:
            wait_for_index_online(driver, vector_index_name, timeout=300)

    # Verify indexes
    print()
    print("Index Status:")
    indexes = verify_indexes(driver)
    for idx in indexes:
        if idx["name"] in [vector_index_name, fulltext_index_name]:
            print(f"  {idx['name']}: {idx['state']} ({idx['type']})")

    print()
    print("=" * 60)
    print("INDEX SETUP COMPLETE")
    print("=" * 60)

    if close_driver:
        driver.close()

    return results


def drop_index(driver=None, index_name: str = None) -> bool:
    """
    Drop an index by name.

    Args:
        driver: Neo4j driver
        index_name: Name of index to drop

    Returns:
        True if dropped successfully
    """
    if not index_name:
        return False

    if driver is None:
        driver = get_neo4j_driver()
        close_driver = True
    else:
        close_driver = False

    try:
        with driver.session() as session:
            session.run(f"DROP INDEX `{index_name}`")
        print(f"Dropped index '{index_name}'")
        success = True
    except Exception as e:
        print(f"Error dropping index: {e}")
        success = False

    if close_driver:
        driver.close()

    return success


if __name__ == "__main__":
    # Setup all indexes
    results = setup_all_indexes()
    print(f"\nResults: {results}")
