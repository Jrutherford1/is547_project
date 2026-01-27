"""
GraphRAG retriever implementation for semantic document search.

This module provides retriever classes for:
- Vector similarity search
- Hybrid search (vector + fulltext)
- Graph-enhanced retrieval with committee/person context
"""

import os
from typing import Optional, Dict, Any, List
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


def get_query_embedding(
    query: str,
    model: str = None
) -> List[float]:
    """
    Generate embedding for a query string.

    Args:
        query: Query text
        model: Embedding model (default: from env)

    Returns:
        Embedding vector as list of floats
    """
    if model is None:
        model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

    response = ollama.embed(model=model, input=query)
    return response["embeddings"][0]


def vector_search(
    query: str,
    driver=None,
    index_name: str = "document_embeddings",
    top_k: int = 5,
    model: str = None
) -> List[Dict[str, Any]]:
    """
    Perform vector similarity search on documents.

    Args:
        query: Search query text
        driver: Neo4j driver
        index_name: Vector index name
        top_k: Number of results to return
        model: Embedding model

    Returns:
        List of matching documents with scores
    """
    if driver is None:
        driver = get_neo4j_driver()
        close_driver = True
    else:
        close_driver = False

    # Get query embedding
    query_embedding = get_query_embedding(query, model)

    # Vector search query
    cypher = """
    CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
    YIELD node, score
    OPTIONAL MATCH (node)-[:BELONGS_TO]->(c:Committee)
    RETURN
        node.id AS id,
        node.name AS name,
        node.dateCreated AS date,
        node.additionalType AS docType,
        c.name AS committee,
        score
    ORDER BY score DESC
    """

    with driver.session() as session:
        result = session.run(
            cypher,
            index_name=index_name,
            top_k=top_k,
            embedding=query_embedding
        )
        documents = [dict(record) for record in result]

    if close_driver:
        driver.close()

    return documents


def hybrid_search(
    query: str,
    driver=None,
    vector_index: str = "document_embeddings",
    fulltext_index: str = "document_fulltext",
    top_k: int = 5,
    model: str = None
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search combining vector and fulltext.

    Args:
        query: Search query text
        driver: Neo4j driver
        vector_index: Vector index name
        fulltext_index: Fulltext index name
        top_k: Number of results to return
        model: Embedding model

    Returns:
        List of matching documents with combined scores
    """
    if driver is None:
        driver = get_neo4j_driver()
        close_driver = True
    else:
        close_driver = False

    # Get query embedding
    query_embedding = get_query_embedding(query, model)

    # Hybrid search: combine vector and fulltext results
    cypher = """
    // Vector search
    CALL db.index.vector.queryNodes($vector_index, $top_k, $embedding)
    YIELD node AS vNode, score AS vScore

    // Get fulltext matches for the same documents
    WITH vNode, vScore
    OPTIONAL MATCH (vNode)
    WHERE vNode.name CONTAINS $query_text OR vNode.description CONTAINS $query_text

    // Calculate combined score
    WITH vNode,
         vScore AS vectorScore,
         CASE WHEN vNode.name CONTAINS $query_text THEN 0.2 ELSE 0 END +
         CASE WHEN vNode.description CONTAINS $query_text THEN 0.1 ELSE 0 END AS textBoost

    // Get committee
    OPTIONAL MATCH (vNode)-[:BELONGS_TO]->(c:Committee)

    RETURN
        vNode.id AS id,
        vNode.name AS name,
        vNode.dateCreated AS date,
        vNode.additionalType AS docType,
        c.name AS committee,
        vectorScore,
        textBoost,
        vectorScore + textBoost AS combinedScore
    ORDER BY combinedScore DESC
    LIMIT $top_k
    """

    with driver.session() as session:
        result = session.run(
            cypher,
            vector_index=vector_index,
            top_k=top_k * 2,  # Get more for reranking
            embedding=query_embedding,
            query_text=query
        )
        documents = [dict(record) for record in result]

    if close_driver:
        driver.close()

    return documents[:top_k]


def graph_enhanced_search(
    query: str,
    driver=None,
    vector_index: str = "document_embeddings",
    top_k: int = 5,
    model: str = None,
    include_people: bool = True,
    include_orgs: bool = True
) -> List[Dict[str, Any]]:
    """
    Perform graph-enhanced search with relationship context.

    Returns documents with related people and organizations from the graph.

    Args:
        query: Search query text
        driver: Neo4j driver
        vector_index: Vector index name
        top_k: Number of results to return
        model: Embedding model
        include_people: Include mentioned people
        include_orgs: Include mentioned organizations

    Returns:
        List of documents with graph context
    """
    if driver is None:
        driver = get_neo4j_driver()
        close_driver = True
    else:
        close_driver = False

    # Get query embedding
    query_embedding = get_query_embedding(query, model)

    # Graph-enhanced query
    cypher = """
    CALL db.index.vector.queryNodes($vector_index, $top_k, $embedding)
    YIELD node, score

    // Get committee
    OPTIONAL MATCH (node)-[:BELONGS_TO]->(c:Committee)

    // Get mentioned people (top 5)
    OPTIONAL MATCH (node)-[:MENTIONS]->(p:Person)
    WITH node, score, c,
         COLLECT(DISTINCT p.name)[0..5] AS people

    // Get mentioned organizations (top 3)
    OPTIONAL MATCH (node)-[:MENTIONS]->(o:Organization)
    WITH node, score, c, people,
         COLLECT(DISTINCT o.name)[0..3] AS orgs

    RETURN
        node.id AS id,
        node.name AS name,
        node.dateCreated AS date,
        node.additionalType AS docType,
        c.name AS committee,
        people AS mentionedPeople,
        orgs AS mentionedOrgs,
        score
    ORDER BY score DESC
    """

    with driver.session() as session:
        result = session.run(
            cypher,
            vector_index=vector_index,
            top_k=top_k,
            embedding=query_embedding
        )
        documents = [dict(record) for record in result]

    if close_driver:
        driver.close()

    return documents


def search_by_person(
    person_name: str,
    driver=None,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Find documents mentioning a specific person.

    Args:
        person_name: Name of person to search for
        driver: Neo4j driver
        top_k: Maximum results

    Returns:
        List of documents mentioning the person
    """
    if driver is None:
        driver = get_neo4j_driver()
        close_driver = True
    else:
        close_driver = False

    cypher = """
    MATCH (p:Person)
    WHERE toLower(p.name) CONTAINS toLower($name)
    MATCH (d:Document)-[:MENTIONS]->(p)
    OPTIONAL MATCH (d)-[:BELONGS_TO]->(c:Committee)
    RETURN
        d.id AS id,
        d.name AS name,
        d.dateCreated AS date,
        d.committee AS committee,
        c.name AS committeeName,
        p.name AS matchedPerson
    ORDER BY d.dateCreated DESC
    LIMIT $top_k
    """

    with driver.session() as session:
        result = session.run(cypher, name=person_name, top_k=top_k)
        documents = [dict(record) for record in result]

    if close_driver:
        driver.close()

    return documents


def search_by_committee(
    committee_name: str,
    driver=None,
    top_k: int = 20
) -> List[Dict[str, Any]]:
    """
    Find documents from a specific committee.

    Args:
        committee_name: Name of committee
        driver: Neo4j driver
        top_k: Maximum results

    Returns:
        List of documents from the committee
    """
    if driver is None:
        driver = get_neo4j_driver()
        close_driver = True
    else:
        close_driver = False

    cypher = """
    MATCH (d:Document)
    WHERE toLower(d.committee) CONTAINS toLower($name)
    OPTIONAL MATCH (d)-[:MENTIONS]->(p:Person)
    WITH d, COLLECT(DISTINCT p.name)[0..5] AS people
    RETURN
        d.id AS id,
        d.name AS name,
        d.dateCreated AS date,
        d.additionalType AS docType,
        d.committee AS committee,
        people AS mentionedPeople
    ORDER BY d.dateCreated DESC
    LIMIT $top_k
    """

    with driver.session() as session:
        result = session.run(cypher, name=committee_name, top_k=top_k)
        documents = [dict(record) for record in result]

    if close_driver:
        driver.close()

    return documents


class GraphRAGRetriever:
    """
    High-level retriever class for GraphRAG operations.

    Combines vector search, fulltext search, and graph traversal.
    """

    def __init__(
        self,
        driver=None,
        vector_index: str = "document_embeddings",
        fulltext_index: str = "document_fulltext",
        embedding_model: str = None
    ):
        """
        Initialize the retriever.

        Args:
            driver: Neo4j driver (creates one if not provided)
            vector_index: Name of vector index
            fulltext_index: Name of fulltext index
            embedding_model: Ollama embedding model name
        """
        self.driver = driver or get_neo4j_driver()
        self.vector_index = vector_index
        self.fulltext_index = fulltext_index
        self.embedding_model = embedding_model or os.getenv(
            "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"
        )
        self._owns_driver = driver is None

    def search(
        self,
        query: str,
        top_k: int = 5,
        search_type: str = "graph_enhanced"
    ) -> List[Dict[str, Any]]:
        """
        Search for documents matching the query.

        Args:
            query: Search query
            top_k: Number of results
            search_type: One of 'vector', 'hybrid', 'graph_enhanced'

        Returns:
            List of matching documents
        """
        if search_type == "vector":
            return vector_search(
                query, self.driver, self.vector_index,
                top_k, self.embedding_model
            )
        elif search_type == "hybrid":
            return hybrid_search(
                query, self.driver, self.vector_index,
                self.fulltext_index, top_k, self.embedding_model
            )
        else:  # graph_enhanced
            return graph_enhanced_search(
                query, self.driver, self.vector_index,
                top_k, self.embedding_model
            )

    def find_by_person(self, person_name: str, top_k: int = 10):
        """Find documents mentioning a person."""
        return search_by_person(person_name, self.driver, top_k)

    def find_by_committee(self, committee_name: str, top_k: int = 20):
        """Find documents from a committee."""
        return search_by_committee(committee_name, self.driver, top_k)

    def close(self):
        """Close the Neo4j driver if we own it."""
        if self._owns_driver and self.driver:
            self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    # Test retriever
    with GraphRAGRetriever() as retriever:
        print("Testing vector search...")
        results = retriever.search("budget discussions", top_k=3)
        for r in results:
            print(f"  {r['name']}: {r.get('score', 'N/A')}")
