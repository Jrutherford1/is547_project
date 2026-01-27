"""
GraphRAG verification and testing utilities.

This module provides functions to verify the GraphRAG setup:
- Ollama connectivity
- Embedding coverage
- Index status
- Retrieval testing
- End-to-end Q&A testing
"""

import os
import time
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

import ollama
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


def verify_ollama_connection(
    host: str = None
) -> Dict[str, Any]:
    """
    Test Ollama API is accessible.

    Args:
        host: Ollama host URL

    Returns:
        Dict with status and available models
    """
    if host is None:
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    result = {
        "status": "unknown",
        "host": host,
        "models": [],
        "error": None
    }

    try:
        # List available models
        models = ollama.list()
        # Handle new ollama library API (uses attributes, not dict)
        if hasattr(models, 'models'):
            result["models"] = [m.model for m in models.models]
        else:
            result["models"] = [m["name"] for m in models.get("models", [])]
        result["status"] = "connected"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def verify_neo4j_connection(driver=None) -> Dict[str, Any]:
    """
    Test Neo4j connection.

    Args:
        driver: Neo4j driver

    Returns:
        Dict with status and database info
    """
    if driver is None:
        driver = get_neo4j_driver()
        close_driver = True
    else:
        close_driver = False

    result = {
        "status": "unknown",
        "uri": os.getenv("NEO4J_URI"),
        "error": None,
        "node_count": 0
    }

    try:
        with driver.session() as session:
            # Simple connectivity test
            count = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
            result["node_count"] = count
            result["status"] = "connected"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    if close_driver:
        driver.close()

    return result


def verify_embeddings_coverage(driver=None) -> Dict[str, Any]:
    """
    Check how many documents have embeddings.

    Args:
        driver: Neo4j driver

    Returns:
        Dict with embedding statistics
    """
    if driver is None:
        driver = get_neo4j_driver()
        close_driver = True
    else:
        close_driver = False

    result = {
        "total_documents": 0,
        "with_embeddings": 0,
        "without_embeddings": 0,
        "coverage_percent": 0.0
    }

    try:
        with driver.session() as session:
            query = """
            MATCH (d:Document)
            RETURN
                count(d) AS total,
                count(d.embedding) AS with_embeddings
            """
            record = session.run(query).single()
            result["total_documents"] = record["total"]
            result["with_embeddings"] = record["with_embeddings"]
            result["without_embeddings"] = record["total"] - record["with_embeddings"]

            if record["total"] > 0:
                result["coverage_percent"] = (record["with_embeddings"] / record["total"]) * 100
    except Exception as e:
        result["error"] = str(e)

    if close_driver:
        driver.close()

    return result


def verify_indexes(driver=None) -> Dict[str, Any]:
    """
    Check index status.

    Args:
        driver: Neo4j driver

    Returns:
        Dict with index information
    """
    if driver is None:
        driver = get_neo4j_driver()
        close_driver = True
    else:
        close_driver = False

    result = {
        "indexes": [],
        "vector_index_online": False,
        "fulltext_index_online": False
    }

    try:
        with driver.session() as session:
            records = session.run("SHOW INDEXES")
            for record in records:
                idx_info = {
                    "name": record["name"],
                    "type": record["type"],
                    "state": record["state"]
                }
                result["indexes"].append(idx_info)

                if record["name"] == "document_embeddings" and record["state"] == "ONLINE":
                    result["vector_index_online"] = True
                if record["name"] == "document_fulltext" and record["state"] == "ONLINE":
                    result["fulltext_index_online"] = True
    except Exception as e:
        result["error"] = str(e)

    if close_driver:
        driver.close()

    return result


def test_embedding_generation(
    model: str = None
) -> Dict[str, Any]:
    """
    Test embedding generation.

    Args:
        model: Embedding model name

    Returns:
        Dict with test results
    """
    if model is None:
        model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

    result = {
        "model": model,
        "status": "unknown",
        "dimensions": 0,
        "latency_ms": 0,
        "error": None
    }

    test_text = "This is a test document about library committee meetings."

    try:
        start = time.time()
        response = ollama.embed(model=model, input=test_text)
        latency = (time.time() - start) * 1000

        embedding = response["embeddings"][0]
        result["dimensions"] = len(embedding)
        result["latency_ms"] = round(latency, 2)
        result["status"] = "success"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def test_vector_search(
    driver=None,
    model: str = None
) -> Dict[str, Any]:
    """
    Test vector search functionality.

    Args:
        driver: Neo4j driver
        model: Embedding model

    Returns:
        Dict with test results
    """
    if driver is None:
        driver = get_neo4j_driver()
        close_driver = True
    else:
        close_driver = False

    if model is None:
        model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

    result = {
        "status": "unknown",
        "results_count": 0,
        "latency_ms": 0,
        "error": None,
        "sample_results": []
    }

    test_query = "budget discussions"

    try:
        # Generate query embedding
        response = ollama.embed(model=model, input=test_query)
        query_embedding = response["embeddings"][0]

        # Run vector search
        start = time.time()
        with driver.session() as session:
            cypher = """
            CALL db.index.vector.queryNodes('document_embeddings', 5, $embedding)
            YIELD node, score
            RETURN node.name AS name, score
            """
            records = session.run(cypher, embedding=query_embedding)
            results = [dict(r) for r in records]

        latency = (time.time() - start) * 1000

        result["results_count"] = len(results)
        result["latency_ms"] = round(latency, 2)
        result["sample_results"] = results[:3]
        result["status"] = "success" if results else "no_results"

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    if close_driver:
        driver.close()

    return result


def test_llm_generation(
    model: str = None
) -> Dict[str, Any]:
    """
    Test LLM response generation.

    Args:
        model: LLM model name

    Returns:
        Dict with test results
    """
    if model is None:
        model = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")

    result = {
        "model": model,
        "status": "unknown",
        "latency_ms": 0,
        "response_length": 0,
        "error": None
    }

    test_prompt = "In one sentence, what is a library committee?"

    try:
        start = time.time()
        response = ollama.generate(
            model=model,
            prompt=test_prompt,
            options={"temperature": 0.1, "num_predict": 100}
        )
        latency = (time.time() - start) * 1000

        result["latency_ms"] = round(latency, 2)
        result["response_length"] = len(response.get("response", ""))
        result["status"] = "success"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def run_full_verification(
    driver=None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run all verification tests.

    Args:
        driver: Neo4j driver
        verbose: Print results as we go

    Returns:
        Dict with all verification results
    """
    if driver is None:
        driver = get_neo4j_driver()
        close_driver = True
    else:
        close_driver = False

    results = {
        "overall_status": "unknown",
        "tests": {}
    }

    if verbose:
        print("=" * 60)
        print("GRAPHRAG VERIFICATION")
        print("=" * 60)
        print()

    # 1. Ollama connection
    if verbose:
        print("1. Testing Ollama connection...")
    results["tests"]["ollama"] = verify_ollama_connection()
    if verbose:
        status = results["tests"]["ollama"]["status"]
        models = results["tests"]["ollama"]["models"]
        print(f"   Status: {status}")
        if models:
            print(f"   Models: {', '.join(models[:5])}")
        print()

    # 2. Neo4j connection
    if verbose:
        print("2. Testing Neo4j connection...")
    results["tests"]["neo4j"] = verify_neo4j_connection(driver)
    if verbose:
        status = results["tests"]["neo4j"]["status"]
        count = results["tests"]["neo4j"]["node_count"]
        print(f"   Status: {status}")
        print(f"   Total nodes: {count}")
        print()

    # 3. Embedding coverage
    if verbose:
        print("3. Checking embedding coverage...")
    results["tests"]["embeddings"] = verify_embeddings_coverage(driver)
    if verbose:
        total = results["tests"]["embeddings"]["total_documents"]
        with_emb = results["tests"]["embeddings"]["with_embeddings"]
        pct = results["tests"]["embeddings"]["coverage_percent"]
        print(f"   Documents: {total}")
        print(f"   With embeddings: {with_emb} ({pct:.1f}%)")
        print()

    # 4. Index status
    if verbose:
        print("4. Checking indexes...")
    results["tests"]["indexes"] = verify_indexes(driver)
    if verbose:
        vec = results["tests"]["indexes"]["vector_index_online"]
        ft = results["tests"]["indexes"]["fulltext_index_online"]
        print(f"   Vector index online: {vec}")
        print(f"   Fulltext index online: {ft}")
        print()

    # 5. Embedding generation test
    if verbose:
        print("5. Testing embedding generation...")
    results["tests"]["embed_test"] = test_embedding_generation()
    if verbose:
        status = results["tests"]["embed_test"]["status"]
        dims = results["tests"]["embed_test"]["dimensions"]
        latency = results["tests"]["embed_test"]["latency_ms"]
        print(f"   Status: {status}")
        print(f"   Dimensions: {dims}")
        print(f"   Latency: {latency}ms")
        print()

    # 6. Vector search test (only if embeddings exist)
    if results["tests"]["embeddings"]["with_embeddings"] > 0:
        if verbose:
            print("6. Testing vector search...")
        results["tests"]["search_test"] = test_vector_search(driver)
        if verbose:
            status = results["tests"]["search_test"]["status"]
            count = results["tests"]["search_test"]["results_count"]
            latency = results["tests"]["search_test"]["latency_ms"]
            print(f"   Status: {status}")
            print(f"   Results: {count}")
            print(f"   Latency: {latency}ms")
            print()
    else:
        results["tests"]["search_test"] = {"status": "skipped", "reason": "no embeddings"}
        if verbose:
            print("6. Skipping vector search test (no embeddings)")
            print()

    # 7. LLM generation test
    if verbose:
        print("7. Testing LLM generation...")
    results["tests"]["llm_test"] = test_llm_generation()
    if verbose:
        status = results["tests"]["llm_test"]["status"]
        latency = results["tests"]["llm_test"]["latency_ms"]
        print(f"   Status: {status}")
        print(f"   Latency: {latency}ms")
        print()

    # Determine overall status
    all_passed = all(
        t.get("status") in ["success", "connected", "skipped"]
        for t in results["tests"].values()
    )
    results["overall_status"] = "passed" if all_passed else "failed"

    if verbose:
        print("=" * 60)
        print(f"OVERALL STATUS: {results['overall_status'].upper()}")
        print("=" * 60)

    if close_driver:
        driver.close()

    return results


if __name__ == "__main__":
    run_full_verification()
