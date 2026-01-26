# Feature Implementation Plan: GraphRAG Ollama Integration

**Feature Name:** `graphrag-ollama-integration`
**Created:** 2026-01-11
**Status:** Planning

---

## 1. Summary

This feature adds local, privacy-preserving GraphRAG capabilities to the IS547 committee documents project using the official `neo4j-graphrag-python` library with Ollama for both embeddings and language model operations. The implementation will enable:

- **Semantic document search** via vector embeddings stored in Neo4j
- **Hybrid retrieval** combining vector similarity + graph traversal
- **Question-answering** about committees, people, and topics using local LLMs
- **Full privacy** - no external API calls, all processing runs locally via Ollama

The feature leverages the existing Neo4j graph (2,198 documents, 1,242 people, 26 committees) and text extraction pipeline, adding a semantic layer on top of the structural knowledge graph.

---

## 2. Codebase Analysis

### Existing Infrastructure

**Neo4j Connection (Validated)**
- URI: `bolt://localhost:7687`
- Credentials stored in `.env` file
- Driver: `neo4j-python-driver` (already used in `neo4j_import.py`)
- Graph populated with 2,198 Document nodes containing `filePath` property

**Text Extraction Pipeline (Ready to Reuse)**
- Module: `data_pipeline/nlp_term_extraction_preview.py`
- Function: `extract_text(file_path)` - handles PDF, DOCX, PPTX, TXT
- Already tested on full dataset during NLP entity extraction phase
- Supports batch processing for efficiency

**Existing Graph Schema**
```cypher
# Nodes
(:Document {id, name, dateCreated, fileFormat, additionalType, filePath, checksum, committee})
(:Person {name, mentionCount})
(:Committee {name})
(:Organization {name, mentionCount})
(:Location {name, mentionCount})

# Relationships
(:Document)-[:BELONGS_TO]->(:Committee)
(:Document)-[:MENTIONS]->(:Person|Organization|Location)
(:Person)-[:CO_APPEARS_WITH {count}]->(:Person)
```

**Dependencies Already Installed**
- `neo4j` - Neo4j Python driver
- `spacy` - NLP (already used for entity extraction)
- `python-dotenv` - Environment variable management
- Text extraction libraries: `pdfplumber`, `python-docx`, `python-pptx`

### Integration Points

1. **Document Nodes** - Add `embedding` property (LIST<FLOAT> or new VECTOR type)
2. **Text Extraction** - Reuse existing `extract_text()` function
3. **Notebook Workflow** - Add new cells to `is547_project.ipynb` after Neo4j import
4. **File Path Access** - Use `Document.filePath` property to retrieve documents for embedding

### Patterns to Follow

The project uses a modular architecture with functions in `data_pipeline/`:
- Each major operation gets its own Python module
- Functions are importable and testable independently
- Progress reporting with counters every 100-500 items
- Batch processing for efficiency (seen in NLP extraction: `batch_size=50`)
- Graceful error handling with try/except and skip-on-error logic
- Environment variables for configuration (`.env` file)

---

## 3. Technical Research Summary

### Neo4j GraphRAG Python Package

The official `neo4j-graphrag-python` library provides:

1. **Embedder Interfaces** - OllamaEmbeddings, SentenceTransformerEmbeddings, etc.
2. **Vector Index Management** - `create_vector_index()`, `upsert_vectors()`
3. **Multiple Retriever Types**:
   - `VectorRetriever` - Pure semantic similarity search
   - `HybridRetriever` - Vector + fulltext index combination
   - `HybridCypherRetriever` - Hybrid search + graph traversal (BEST FOR THIS PROJECT)
   - `VectorCypherRetriever` - Vector search then custom Cypher queries
4. **LLM Integration** - OllamaLLM, ChatOllama (via LangChain)
5. **GraphRAG Pipeline** - Combines retriever + LLM for question answering

**Key Advantage:** The library handles all the complexity of vector search, score normalization, and graph integration with simple Python APIs.

### Ollama Configuration

**Recommended Models for Local Use:**

| Model Type | Model Name | Dimensions | Context | Best For |
|------------|-----------|------------|---------|----------|
| Embeddings | `nomic-embed-text` | 768 | 8,192 tokens | General text, outperforms OpenAI ada-002 |
| Embeddings | `all-minilm` | 384 | 512 tokens | Fast, efficient, smaller vectors |
| Embeddings | `mxbai-embed-large` | 1,024 | 512 tokens | High accuracy |
| LLM | `llama3.2` | - | 128K tokens | Latest Llama, good reasoning |
| LLM | `mistral` | - | 32K tokens | Fast inference, good quality |
| LLM | `orca-mini` | - | 2K tokens | Lightweight, fast responses |

**Recommendation for this project:**
- **Embeddings:** `nomic-embed-text` (768 dimensions, strong performance, 8K context handles long docs)
- **LLM:** `llama3.2` or `mistral` (both have good reasoning for document Q&A)

**Ollama Installation:**
```bash
# Install Ollama (if not already installed)
curl https://ollama.ai/install.sh | sh

# Pull models
ollama pull nomic-embed-text
ollama pull llama3.2
```

### Neo4j Vector Index Requirements

**Neo4j 5.x Vector Index Syntax:**
```cypher
CREATE VECTOR INDEX `document_embeddings`
FOR (d:Document) ON (d.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 768,
    `vector.similarity_function`: 'cosine'
  }
}
```

**Important Notes:**
- Embeddings stored as `LIST<FLOAT>` property on nodes
- Approximate nearest neighbor algorithm (trade accuracy for speed)
- Cosine similarity recommended for text embeddings
- Index creation is asynchronous (may take time for 2,198 docs)

### Hybrid Retrieval Strategy

The **HybridCypherRetriever** is ideal for this project because it:
1. **Semantic search** via vector embeddings (finds conceptually similar docs)
2. **Keyword matching** via fulltext index (catches exact terms, dates, names)
3. **Graph enrichment** via custom Cypher (adds committee context, related people)

**Example Query Pattern:**
```cypher
RETURN
  node.name AS documentName,
  node.dateCreated AS date,
  node.additionalType AS docType,
  [(node)-[:BELONGS_TO]->(c:Committee) | c.name][0] AS committee,
  [(node)-[:MENTIONS]->(p:Person) | p.name] AS mentionedPeople,
  score AS similarityScore
```

This retrieval query would return documents with their committee context and mentioned people, enabling rich RAG responses.

---

## 4. Implementation Plan

### Phase 1: Environment Setup & Dependencies

**File:** `requirements.txt` (update)

Add:
```txt
neo4j-graphrag[ollama]~=0.7.0
ollama~=0.4.0
sentence-transformers~=3.3.0  # Optional: for local embeddings without Ollama
```

**Note:** The `[ollama]` extra installs Ollama integration dependencies automatically.

**Installation command:**
```bash
pip install "neo4j-graphrag[ollama]"
```

**File:** `.env` (update)

Add:
```env
# GraphRAG Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_LLM_MODEL=llama3.2
EMBEDDING_DIMENSIONS=768
```

### Phase 2: Embedding Generation Module

**File:** `data_pipeline/generate_embeddings.py` (NEW)

**Purpose:** Generate embeddings for all documents and store in Neo4j

**Key Functions:**

1. `setup_embedder()` - Initialize OllamaEmbeddings
   ```python
   from neo4j_graphrag.embeddings import OllamaEmbeddings

   def setup_embedder(model: str = "nomic-embed-text", host: str = "http://localhost:11434"):
       """Initialize Ollama embedder."""
       return OllamaEmbeddings(model_name=model, base_url=host)
   ```

2. `extract_and_embed_documents()` - Main embedding generation loop
   ```python
   def extract_and_embed_documents(
       neo4j_driver,
       embedder,
       base_dir: str = "data/Processed_Committees",
       batch_size: int = 50,
       skip_existing: bool = True,
       limit: Optional[int] = None
   ):
       """
       Extract text from documents, generate embeddings, store in Neo4j.

       Process:
       1. Query Neo4j for Document nodes without embeddings (or all if not skipping)
       2. For each document, use Document.filePath to locate file
       3. Extract text using existing extract_text() function
       4. Generate embeddings in batches for efficiency
       5. Update Document nodes with embedding property
       6. Report progress every 100 documents
       """
   ```

3. `check_embedding_status()` - Verify embedding coverage
   ```python
   def check_embedding_status(neo4j_driver):
       """Query Neo4j to count documents with/without embeddings."""
   ```

**Implementation Details:**

- **Batch Processing:** Process documents in batches of 50 for embedding generation
- **Text Truncation:** Nomic-embed-text supports 8,192 tokens; truncate if needed
- **Error Handling:** Skip documents that fail text extraction, log errors
- **Progress Tracking:** Print status every 100 documents processed
- **Incremental Updates:** Check if `d.embedding` exists before processing (when `skip_existing=True`)

**Text Extraction Integration:**
```python
from data_pipeline.nlp_term_extraction_preview import extract_text

# In processing loop:
text = extract_text(doc_file_path)
if text and len(text.strip()) > 100:  # Minimum viable text
    embedding = embedder.embed_query(text[:8000])  # Truncate to safe length
    # Store in Neo4j
```

### Phase 3: Neo4j Vector Index Setup

**File:** `data_pipeline/setup_vector_index.py` (NEW)

**Purpose:** Create and manage vector indexes and fulltext indexes in Neo4j

**Key Functions:**

1. `create_vector_index()` - Create the vector index
   ```python
   def create_document_vector_index(
       driver,
       index_name: str = "document_embeddings",
       dimensions: int = 768,
       similarity_function: str = "cosine"
   ):
       """
       Create vector index on Document.embedding property.

       Uses neo4j-graphrag-python's create_vector_index() helper.
       """
       from neo4j_graphrag.indexes import create_vector_index

       create_vector_index(
           driver,
           index_name=index_name,
           label="Document",
           embedding_property="embedding",
           dimensions=dimensions,
           similarity_fn=similarity_function
       )
   ```

2. `create_fulltext_index()` - Create fulltext index for hybrid search
   ```python
   def create_document_fulltext_index(
       driver,
       index_name: str = "document_fulltext"
   ):
       """
       Create fulltext index on Document text properties.

       Indexes: name, description, additionalType for keyword matching.
       """
       with driver.session() as session:
           session.run("""
               CREATE FULLTEXT INDEX document_fulltext IF NOT EXISTS
               FOR (d:Document)
               ON EACH [d.name, d.description, d.additionalType]
           """)
   ```

3. `verify_indexes()` - Check index status
   ```python
   def verify_indexes(driver):
       """Query Neo4j to list all indexes and their status."""
       with driver.session() as session:
           result = session.run("SHOW INDEXES")
           return [dict(record) for record in result]
   ```

4. `wait_for_index_online()` - Wait for index to be ready
   ```python
   def wait_for_index_online(driver, index_name: str, timeout: int = 300):
       """
       Poll Neo4j until index state is ONLINE.
       Vector indexes may take minutes to build for 2,198 documents.
       """
   ```

**Execution Order:**
1. Create fulltext index first (faster)
2. Create vector index
3. Wait for both to reach ONLINE state
4. Verify with `SHOW INDEXES`

### Phase 4: Retriever Implementation

**File:** `data_pipeline/graphrag_retriever.py` (NEW)

**Purpose:** Implement retrievers for semantic + graph-enhanced search

**Key Functions:**

1. `create_vector_retriever()` - Basic semantic search
   ```python
   from neo4j_graphrag.retrievers import VectorRetriever

   def create_vector_retriever(driver, embedder, index_name: str = "document_embeddings"):
       """
       Create simple vector similarity retriever.

       Returns documents most semantically similar to query.
       """
       return VectorRetriever(
           driver=driver,
           index_name=index_name,
           embedder=embedder,
           return_properties=["name", "dateCreated", "additionalType", "committee"]
       )
   ```

2. `create_hybrid_cypher_retriever()` - Advanced graph-enhanced retrieval (RECOMMENDED)
   ```python
   from neo4j_graphrag.retrievers import HybridCypherRetriever

   def create_hybrid_cypher_retriever(
       driver,
       embedder,
       vector_index: str = "document_embeddings",
       fulltext_index: str = "document_fulltext"
   ):
       """
       Create hybrid retriever combining:
       - Vector similarity search
       - Fulltext keyword matching
       - Graph traversal for context

       Best for committee document Q&A.
       """

       # Custom Cypher to enrich results with graph context
       retrieval_query = """
       RETURN
         node.name AS documentName,
         node.dateCreated AS documentDate,
         node.additionalType AS documentType,
         node.description AS description,
         [(node)-[:BELONGS_TO]->(c:Committee) | c.name][0] AS committee,
         [(node)-[:MENTIONS]->(p:Person) | p.name][0..5] AS mentionedPeople,
         [(node)-[:MENTIONS]->(o:Organization) | o.name][0..3] AS mentionedOrgs,
         score AS relevanceScore
       """

       return HybridCypherRetriever(
           driver=driver,
           vector_index_name=vector_index,
           fulltext_index_name=fulltext_index,
           retrieval_query=retrieval_query,
           embedder=embedder
       )
   ```

3. `search_documents()` - Convenience wrapper
   ```python
   def search_documents(
       retriever,
       query: str,
       top_k: int = 5,
       filters: Optional[Dict] = None
   ):
       """
       Execute search with optional filters.

       Example filters:
         {"committee": {"$eq": "Executive Committee"}}
         {"documentDate": {"$gte": "2020-01-01"}}
       """
       return retriever.search(
           query_text=query,
           top_k=top_k,
           filters=filters
       )
   ```

**Retriever Selection Guide:**

| Use Case | Retriever | Reason |
|----------|-----------|--------|
| "Find documents about budget discussions" | HybridCypherRetriever | Hybrid search + committee context |
| "What did John Wilkin work on?" | HybridCypherRetriever | Needs person mentions + docs |
| "Documents from 2023 about digital preservation" | HybridCypherRetriever | Date + topic + context |
| Simple similarity only | VectorRetriever | Fastest, no graph traversal |

**Recommendation:** Default to `HybridCypherRetriever` for this project.

### Phase 5: RAG Question-Answering Module

**File:** `data_pipeline/graphrag_qa.py` (NEW)

**Purpose:** Enable natural language questions about committee documents

**Key Functions:**

1. `create_ollama_llm()` - Initialize local LLM
   ```python
   from neo4j_graphrag.llm import OllamaLLM

   def create_ollama_llm(
       model_name: str = "llama3.2",
       host: str = "http://localhost:11434",
       temperature: float = 0.1
   ):
       """
       Initialize Ollama LLM for question answering.

       Low temperature (0.1) for factual, consistent responses.
       """
       return OllamaLLM(
           model_name=model_name,
           model_params={"options": {"temperature": temperature}},
           base_url=host
       )
   ```

2. `create_rag_pipeline()` - Assemble GraphRAG system
   ```python
   from neo4j_graphrag.generation import GraphRAG, RagTemplate

   def create_rag_pipeline(retriever, llm, use_custom_prompt: bool = True):
       """
       Create GraphRAG pipeline combining retriever + LLM.

       Custom prompt optimized for committee document context.
       """

       if use_custom_prompt:
           prompt = RagTemplate(
               prompt="""You are an assistant analyzing library committee documents.

               Use the following context to answer the question. Include specific details like:
               - Committee names
               - Document dates
               - People mentioned
               - Document types (Minutes, Agenda, etc.)

               If you're unsure, say so. Only use information from the provided context.

               Context:
               {context}

               Question: {query_text}

               Answer:""",
               expected_inputs=["context", "query_text"]
           )
           return GraphRAG(retriever=retriever, llm=llm, prompt_template=prompt)
       else:
           return GraphRAG(retriever=retriever, llm=llm)
   ```

3. `ask_question()` - Main Q&A interface
   ```python
   def ask_question(
       rag_pipeline,
       question: str,
       top_k: int = 5,
       return_context: bool = True,
       filters: Optional[Dict] = None
   ):
       """
       Ask a question about committee documents.

       Args:
           rag_pipeline: GraphRAG instance
           question: Natural language question
           top_k: Number of documents to retrieve
           return_context: Whether to return source documents
           filters: Optional Neo4j filters (e.g., date range, committee)

       Returns:
           Dict with 'answer', 'sources', and optionally 'context'
       """
       retriever_config = {"top_k": top_k}
       if filters:
           retriever_config["filters"] = filters

       response = rag_pipeline.search(
           query_text=question,
           retriever_config=retriever_config,
           return_context=return_context
       )

       # Format response
       result = {
           "answer": response.answer,
           "sources": []
       }

       if return_context and hasattr(response, 'retriever_result'):
           for item in response.retriever_result.items:
               result["sources"].append({
                   "document": item.content.get("documentName"),
                   "date": item.content.get("documentDate"),
                   "committee": item.content.get("committee"),
                   "score": item.metadata.get("relevanceScore")
               })

       return result
   ```

4. `example_questions()` - Pre-defined queries for testing
   ```python
   def example_questions():
       """Return list of example questions users can ask."""
       return [
           "What topics were discussed in Executive Committee meetings in 2023?",
           "Which committees has John Wilkin participated in?",
           "What decisions were made about digital preservation?",
           "Show me documents about budget planning from 2022-2024",
           "What are the key initiatives mentioned in recent agendas?",
           "Which people appear most frequently across committees?",
           "What organizations are mentioned in diversity committee documents?"
       ]
   ```

**Response Format Example:**
```python
{
  "answer": "In 2023, the Executive Committee discussed...",
  "sources": [
    {
      "document": "Executive_Committee_Minutes_2023-05-15.pdf",
      "date": "2023-05-15",
      "committee": "Executive Committee",
      "score": 0.89
    },
    ...
  ]
}
```

### Phase 6: Notebook Integration

**File:** `is547_project.ipynb` (ADD NEW CELLS)

**Location:** After the Neo4j import cell (currently cell 50)

**New Cells to Add:**

**Cell 1: GraphRAG Setup Overview (Markdown)**
```markdown
## GraphRAG Integration with Ollama

This section adds semantic search and question-answering capabilities using:
- **neo4j-graphrag-python**: Official Neo4j GraphRAG library
- **Ollama**: Local LLM and embeddings (privacy-preserving)
- **Vector Index**: Semantic similarity search
- **Hybrid Retrieval**: Combines vector search + keyword matching + graph traversal

Prerequisites:
1. Ollama installed and running (`ollama serve`)
2. Models downloaded: `ollama pull nomic-embed-text` and `ollama pull llama3.2`
3. Neo4j database populated (run previous cells first)
```

**Cell 2: Generate Embeddings (Code)**
```python
# Step 1: Generate embeddings for all documents
from data_pipeline.generate_embeddings import (
    setup_embedder,
    extract_and_embed_documents,
    check_embedding_status
)
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

# Connect to Neo4j
driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI'),
    auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
)

# Setup Ollama embedder
embedder = setup_embedder(
    model=os.getenv('OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text'),
    host=os.getenv('OLLAMA_HOST', 'http://localhost:11434')
)

# Generate embeddings (skip if already done)
extract_and_embed_documents(
    neo4j_driver=driver,
    embedder=embedder,
    batch_size=50,
    skip_existing=True,  # Set to False to regenerate all
    limit=None  # Process all documents
)

# Check status
check_embedding_status(driver)
```

**Cell 3: Create Vector Indexes (Code)**
```python
# Step 2: Create vector and fulltext indexes
from data_pipeline.setup_vector_index import (
    create_document_vector_index,
    create_document_fulltext_index,
    wait_for_index_online,
    verify_indexes
)

# Create fulltext index (for hybrid search)
print("Creating fulltext index...")
create_document_fulltext_index(driver, index_name="document_fulltext")

# Create vector index
print("Creating vector index...")
create_document_vector_index(
    driver,
    index_name="document_embeddings",
    dimensions=int(os.getenv('EMBEDDING_DIMENSIONS', 768)),
    similarity_function="cosine"
)

# Wait for indexes to be ready
print("\nWaiting for indexes to come online...")
wait_for_index_online(driver, "document_embeddings", timeout=300)
wait_for_index_online(driver, "document_fulltext", timeout=60)

# Verify
print("\nIndex Status:")
for idx in verify_indexes(driver):
    print(f"  {idx['name']}: {idx['state']} ({idx['type']})")
```

**Cell 4: Test Semantic Search (Code)**
```python
# Step 3: Test vector retrieval
from data_pipeline.graphrag_retriever import (
    create_hybrid_cypher_retriever,
    search_documents
)

# Create retriever
retriever = create_hybrid_cypher_retriever(
    driver=driver,
    embedder=embedder,
    vector_index="document_embeddings",
    fulltext_index="document_fulltext"
)

# Test query
test_query = "What topics were discussed in Executive Committee meetings?"
results = search_documents(retriever, test_query, top_k=5)

print(f"Query: {test_query}\n")
for item in results.items:
    print(f"Document: {item.content.get('documentName')}")
    print(f"Date: {item.content.get('documentDate')}")
    print(f"Committee: {item.content.get('committee')}")
    print(f"Score: {item.metadata.get('relevanceScore'):.3f}")
    print(f"People: {item.content.get('mentionedPeople', [])[:3]}")
    print("-" * 80)
```

**Cell 5: Setup Question Answering (Code)**
```python
# Step 4: Initialize RAG pipeline for Q&A
from data_pipeline.graphrag_qa import (
    create_ollama_llm,
    create_rag_pipeline,
    ask_question,
    example_questions
)

# Create LLM
llm = create_ollama_llm(
    model_name=os.getenv('OLLAMA_LLM_MODEL', 'llama3.2'),
    host=os.getenv('OLLAMA_HOST', 'http://localhost:11434'),
    temperature=0.1  # Low temp for factual answers
)

# Create RAG pipeline
rag = create_rag_pipeline(retriever, llm, use_custom_prompt=True)

print("GraphRAG Question-Answering System Ready!")
print("\nExample Questions:")
for i, q in enumerate(example_questions(), 1):
    print(f"{i}. {q}")
```

**Cell 6: Ask Questions (Code)**
```python
# Step 5: Ask questions about committee documents
question = "What were the main initiatives discussed in 2023?"

response = ask_question(
    rag_pipeline=rag,
    question=question,
    top_k=5,
    return_context=True
)

print(f"Question: {question}\n")
print(f"Answer:\n{response['answer']}\n")
print(f"\nSources ({len(response['sources'])} documents):")
for i, src in enumerate(response['sources'], 1):
    print(f"{i}. {src['document']} ({src['date']}) - Score: {src['score']:.3f}")
    print(f"   Committee: {src['committee']}")
```

**Cell 7: Interactive Q&A (Code)**
```python
# Step 6: Interactive question answering
def interactive_qa():
    """Run interactive Q&A session."""
    print("Interactive GraphRAG Q&A (type 'quit' to exit)\n")

    while True:
        question = input("Your question: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            break

        if not question:
            continue

        print("\nThinking...\n")

        response = ask_question(
            rag_pipeline=rag,
            question=question,
            top_k=5,
            return_context=True
        )

        print(f"Answer:\n{response['answer']}\n")
        print(f"Sources: {len(response['sources'])} documents\n")
        print("-" * 80 + "\n")

# Run interactive mode (comment out if not needed)
# interactive_qa()

# Or ask single questions:
questions_to_ask = [
    "Who are the most frequently mentioned people in committee documents?",
    "What topics were discussed about digital preservation?",
    "Which committees worked on diversity initiatives?"
]

for q in questions_to_ask:
    print(f"\nQ: {q}")
    resp = ask_question(rag, q, top_k=3, return_context=False)
    print(f"A: {resp['answer']}\n")
```

**Cell 8: Filtered Search Example (Code)**
```python
# Step 7: Advanced queries with filters
from datetime import datetime

# Example: Only search Executive Committee docs from 2023
filtered_response = ask_question(
    rag_pipeline=rag,
    question="What budget decisions were made?",
    top_k=5,
    filters={
        "committee": {"$eq": "Executive Committee"},
        "documentDate": {
            "$gte": "2023-01-01",
            "$lte": "2023-12-31"
        }
    },
    return_context=True
)

print("Filtered Query: Executive Committee budget decisions in 2023")
print(f"\nAnswer:\n{filtered_response['answer']}\n")
print(f"Sources: {len(filtered_response['sources'])} matching documents")
```

### Phase 7: Verification & Testing Module

**File:** `data_pipeline/graphrag_verification.py` (NEW)

**Purpose:** Validate GraphRAG setup and provide diagnostic tools

**Key Functions:**

1. `verify_ollama_connection()` - Check Ollama is running
   ```python
   def verify_ollama_connection(host: str = "http://localhost:11434"):
       """Test Ollama API is accessible."""
       import requests
       try:
           response = requests.get(f"{host}/api/tags")
           return response.status_code == 200
       except:
           return False
   ```

2. `verify_embeddings_coverage()` - Check embedding completeness
   ```python
   def verify_embeddings_coverage(driver):
       """
       Check how many documents have embeddings.

       Returns:
           dict with total, with_embeddings, without_embeddings counts
       """
   ```

3. `test_vector_search()` - Validate vector index works
   ```python
   def test_vector_search(driver, embedder, index_name: str = "document_embeddings"):
       """
       Execute test vector query to confirm index is functional.
       """
   ```

4. `benchmark_retrieval()` - Measure retrieval performance
   ```python
   def benchmark_retrieval(retriever, test_queries: List[str]):
       """
       Run test queries and measure retrieval speed.

       Returns timing and result count statistics.
       """
   ```

5. `run_full_verification()` - Comprehensive system check
   ```python
   def run_full_verification(driver, embedder, retriever, llm):
       """
       Run all verification tests and print report.

       Checks:
       - Ollama connectivity
       - Embedding coverage
       - Vector index functionality
       - Retriever operation
       - LLM response generation
       """
   ```

---

## 5. Integration Points & Data Flow

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Existing Pipeline                            │
│  Documents → Text Extraction → NLP → Entities → Neo4j Graph     │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   NEW: GraphRAG Pipeline                         │
│                                                                   │
│  1. Document.filePath → extract_text() → Document Text           │
│                                                                   │
│  2. Document Text → Ollama nomic-embed-text → Embeddings (768d)  │
│                                                                   │
│  3. Embeddings → Neo4j Document.embedding property               │
│                                                                   │
│  4. CREATE VECTOR INDEX on Document.embedding                    │
│                                                                   │
│  5. CREATE FULLTEXT INDEX on Document text properties            │
│                                                                   │
│  6. User Query → Ollama Embedding → Vector + Fulltext Search     │
│                           ↓                                       │
│                    Similar Documents                              │
│                           ↓                                       │
│                  Graph Traversal (Cypher)                         │
│                           ↓                                       │
│              Enriched Context (docs + committees + people)        │
│                           ↓                                       │
│                  Ollama LLM (llama3.2)                            │
│                           ↓                                       │
│                    Natural Language Answer                        │
└─────────────────────────────────────────────────────────────────┘
```

### Neo4j Schema Changes

**Before:**
```cypher
(:Document {id, name, dateCreated, fileFormat, additionalType, filePath, checksum, committee})
```

**After:**
```cypher
(:Document {
  id,
  name,
  dateCreated,
  fileFormat,
  additionalType,
  filePath,
  checksum,
  committee,
  embedding: LIST<FLOAT>  // NEW: 768-dimensional vector
})

// NEW Indexes:
VECTOR INDEX document_embeddings ON Document.embedding (768d, cosine)
FULLTEXT INDEX document_fulltext ON Document [name, description, additionalType]
```

### Integration with Existing Code

**Reused Components:**

1. **Text Extraction** - `nlp_term_extraction_preview.extract_text()`
   - Already handles PDF, DOCX, PPTX, TXT
   - No modifications needed
   - Import and use as-is

2. **Neo4j Connection** - `.env` credentials, driver pattern
   - Same connection parameters
   - Same driver instantiation pattern
   - No changes to existing code

3. **Progress Reporting** - Print every N items pattern
   - Follow existing pattern: `if count % 100 == 0: print(...)`
   - Maintain consistency with rest of pipeline

4. **Batch Processing** - Process files in chunks
   - Use `batch_size=50` like NLP extraction
   - Leverage spaCy's `pipe()` pattern for efficiency

**New Components:**

- All GraphRAG modules are additive (no modifications to existing files)
- New cells added to notebook after Neo4j import
- New dependencies in `requirements.txt`
- New config values in `.env`

### Error Handling Strategy

Follow existing patterns from `add_nlp_terms_to_metadata.py`:

```python
try:
    text = extract_text(file_path)
    if text and len(text.strip()) > 100:
        embedding = embedder.embed_query(text)
        # Store embedding
        processed_count += 1
    else:
        skipped_count += 1
except Exception as e:
    error_count += 1
    print(f"  Error processing {file_path}: {e}")
    continue  # Skip to next document

if processed_count % 100 == 0:
    print(f"  Processed {processed_count} documents...")
```

---

## 6. Considerations & Best Practices

### Performance Considerations

**Embedding Generation Time:**
- 2,198 documents × ~2 seconds each = ~73 minutes total
- Use batch processing to reduce overhead
- Run during off-hours or in background
- Can be interrupted and resumed (skip_existing=True)

**Vector Index Build Time:**
- Initial index creation for 2,198 docs: ~2-5 minutes
- Index must be ONLINE before queries work
- Use `wait_for_index_online()` to ensure readiness

**Query Performance:**
- Vector search: ~100-500ms for top-k=5
- Hybrid search: ~200-800ms (vector + fulltext + graph)
- LLM response: ~2-10 seconds (depends on model, length)
- **Total RAG query time: ~3-11 seconds**

**Optimization Strategies:**
1. Use smaller embedding dimensions (384 for all-minilm vs 768 for nomic)
2. Reduce top_k (3 instead of 5)
3. Use lighter LLM (orca-mini instead of llama3.2)
4. Cache frequent queries (add Redis if needed in future)

### Privacy & Security

**Advantages of Local Setup:**
- All processing happens locally (Ollama + Neo4j on localhost)
- No data sent to external APIs
- No API keys required
- No usage costs
- Full control over models and data

**Security Best Practices:**
- Neo4j credentials remain in `.env` (already gitignored)
- Ollama runs on localhost only (not exposed externally)
- Document embeddings stay in local Neo4j database
- Consider encrypting Neo4j data directory for sensitive content

### Scalability Considerations

**Current Dataset: 2,198 documents**
- Vector index: ~7 MB storage (2,198 × 768 × 4 bytes)
- Manageable on consumer hardware
- Ollama embedding generation: CPU-bound, ~1-2 hours total

**Future Growth: 10,000+ documents**
- Consider sentence-level chunking (instead of full-document embeddings)
- Use external vector DB (Qdrant, Weaviate) for better performance
- Implement incremental embedding updates
- Add caching layer for frequent queries

**Hardware Requirements:**
- Minimum: 8GB RAM, 4 CPU cores, 10GB disk space
- Recommended: 16GB RAM, 8 CPU cores, 20GB disk space
- GPU optional (Ollama can use CUDA/Metal for faster inference)

### Alternative Approaches Considered

**1. Sentence-Level Chunking**
- **Approach:** Split documents into paragraphs/sentences, embed separately
- **Pros:** More granular retrieval, better for long documents
- **Cons:** Complexity, 10x+ more embeddings to manage
- **Decision:** Start with document-level, add chunking if needed

**2. External Vector Database (Weaviate/Qdrant)**
- **Approach:** Store embeddings outside Neo4j
- **Pros:** Better vector search performance, scalability
- **Cons:** Additional infrastructure, data sync complexity
- **Decision:** Use Neo4j native vectors for simplicity, migrate if scale demands

**3. OpenAI API Instead of Ollama**
- **Approach:** Use gpt-4 and text-embedding-3-large
- **Pros:** Higher quality, faster
- **Cons:** Costs money, sends data externally, requires API key
- **Decision:** Ollama for privacy and cost-free operation

**4. Sentence-Transformers Directly (No Ollama)**
- **Approach:** Use sentence-transformers library, skip Ollama
- **Pros:** One less dependency
- **Cons:** Neo4j-graphrag-python integrates well with Ollama
- **Decision:** Use Ollama for consistency with LLM

### Testing Strategy

**Unit Tests (Optional):**
- Test embedder initialization
- Test vector index creation
- Test retriever search
- Test LLM response generation

**Integration Tests:**
1. Generate embeddings for 10 sample documents
2. Create vector index
3. Execute test queries, verify results returned
4. Ask sample question, verify LLM responds

**Validation Queries:**
```python
# Test queries to validate system works:
test_cases = [
    ("Executive Committee", "Should return Executive Committee docs"),
    ("John Wilkin", "Should find docs mentioning this person"),
    ("2023 minutes", "Should find 2023 meeting minutes"),
    ("budget planning", "Should find budget-related documents"),
]
```

**Success Criteria:**
- ✓ All 2,198 documents have embeddings
- ✓ Vector index is ONLINE
- ✓ Test query returns relevant results
- ✓ LLM generates coherent answer citing sources
- ✓ No errors in embedding generation
- ✓ Retrieval time < 1 second
- ✓ RAG response time < 15 seconds

### Potential Challenges & Mitigations

**Challenge 1: Ollama Not Installed/Running**
- **Mitigation:** Add verification step in notebook
- **Solution:** `verify_ollama_connection()` function with clear error message

**Challenge 2: Neo4j Vector Index Fails to Create**
- **Mitigation:** Catch exceptions, provide troubleshooting steps
- **Solution:** Check Neo4j version >= 5.0, ensure sufficient disk space

**Challenge 3: Text Extraction Fails for Some Documents**
- **Mitigation:** Already handled by existing `extract_text()` error handling
- **Solution:** Skip failed documents, log errors, continue processing

**Challenge 4: Embedding Generation Takes Too Long**
- **Mitigation:** Implement batch processing, progress reporting
- **Solution:** Allow interruption/resumption with `skip_existing=True`

**Challenge 5: LLM Responses are Low Quality**
- **Mitigation:** Test multiple models (llama3.2, mistral)
- **Solution:** Provide model selection parameter, document best performers

**Challenge 6: Retrieval Returns Irrelevant Documents**
- **Mitigation:** Use hybrid search (vector + fulltext)
- **Solution:** Tune top_k parameter, add date/committee filters

### Future Enhancement Opportunities

**Phase 2 Features (Not in Initial Implementation):**

1. **Document Chunking**
   - Split long documents into paragraphs
   - Embed at paragraph level for precision
   - Update retrieval to aggregate chunks

2. **Faceted Search UI**
   - Build web interface (Flask/FastAPI)
   - Add filters for committee, date, document type
   - Display results with highlighting

3. **Query History & Analytics**
   - Store user queries in database
   - Track popular questions
   - Identify gaps in document coverage

4. **Multi-hop Reasoning**
   - Connect information across multiple documents
   - Answer comparative questions ("How did policies change over time?")
   - Use graph traversal for complex queries

5. **Automatic Summarization**
   - Generate executive summaries of documents
   - Create committee-level overview reports
   - Periodic digest emails

6. **Integration with Existing Visualization**
   - Add semantic search to knowledge_graph_explorer.html
   - Click person node → show semantically related docs
   - Visual clustering by topic

---

## 7. Example Queries & Use Cases

### Use Case 1: Historical Research

**Question:** "What were the major initiatives discussed by the Executive Committee in 2022?"

**Retrieval Process:**
1. Embed question with nomic-embed-text
2. Vector search finds similar documents
3. Fulltext search catches "Executive Committee" and "2022"
4. Graph traversal confirms committee relationship
5. Return top 5 most relevant documents

**LLM Response Example:**
```
In 2022, the Executive Committee discussed several major initiatives:

1. Digital Preservation Strategy (mentioned in minutes from 2022-03-15)
   - Implementation of new archival system
   - Partnership with HathiTrust

2. Budget Reallocation (minutes from 2022-06-20)
   - 15% increase for digital resources
   - Staffing adjustments

3. Library Space Renovation (agenda from 2022-09-12)
   - Plans for 1st floor redesign
   - Student study space expansion

Sources: 5 documents from Executive Committee between 2022-01-10 and 2022-11-18
```

### Use Case 2: People-Centric Search

**Question:** "Which committees has Sarah Johnson participated in?"

**Retrieval Process:**
1. Vector + fulltext search finds "Sarah Johnson" mentions
2. Graph traversal via `(doc)-[:MENTIONS]->(:Person {name: "Sarah Johnson"})`
3. Aggregate committee names via `(doc)-[:BELONGS_TO]->(c:Committee)`

**Response:**
```
Sarah Johnson has participated in the following committees:

1. Diversity, Equity, Inclusion, and Accessibility (DEIA) Task Force
   - Mentioned in 8 documents between 2021-2023
   - Role: Task force member

2. Outreach and Engagement Committee
   - Mentioned in 5 documents from 2020-2022
   - Collaborated with Maria Garcia and Tom Chen

3. Research and Publication Committee
   - 3 mentions in 2023 documents
   - Recent involvement in publication policy updates

Total: 16 document mentions across 3 committees
```

### Use Case 3: Topic Discovery

**Question:** "What has been discussed about artificial intelligence and machine learning?"

**Retrieval Process:**
1. Semantic search finds conceptually related terms (AI, ML, automation, algorithms)
2. Hybrid search catches exact phrases
3. Return documents spanning multiple committees

**Response:**
```
Artificial intelligence and machine learning have been discussed in several contexts:

1. Research Technologies (2023-2024)
   - Implementation of AI-assisted cataloging
   - Evaluation of ChatGPT for reference services
   - Mentioned in CAPT Committee minutes

2. Digital Scholarship Initiatives (2022-2023)
   - Grant proposals for ML research support
   - Partnership with Computer Science department
   - Discussed in Research Support Task Force

3. Privacy Concerns (2023)
   - DEIA Task Force raised questions about algorithmic bias
   - Policy recommendations for AI tool usage

The discussions span 12 documents across 5 committees, showing growing organizational interest in AI technologies.
```

### Use Case 4: Policy Tracking

**Question:** "How have data privacy policies evolved in committee discussions?"

**Retrieval Process:**
1. Find documents mentioning "privacy", "data protection", "GDPR", "confidentiality"
2. Sort by date to show chronological progression
3. Identify policy documents vs discussion minutes

**Response:**
```
Data privacy policy discussions have evolved through several phases:

Early Awareness (2018-2019):
- Initial mentions in Administrative Council about GDPR compliance
- Privacy concerns for user analytics raised in User Education Committee

Policy Development (2020-2021):
- Privacy Policies Implementation Team formed (2020-05)
- Draft policies circulated across multiple committees
- 8 documents showing collaborative policy development

Implementation (2022-2023):
- Revised policies adopted (2022-03)
- Training requirements established for staff
- Monitoring and compliance procedures documented

Recent Updates (2024):
- AI tool usage guidelines added
- Student data protection protocols enhanced

Key contributors: Multiple committees coordinated efforts, with primary work by Privacy Policies Implementation Team.
```

### Use Case 5: Meeting Preparation

**Question:** "What should I know before attending tomorrow's Budget Group meeting?"

**Retrieval Process:**
1. Filter by committee="Budget Group"
2. Retrieve most recent documents (past 6 months)
3. Identify recurring topics and action items

**Response:**
```
Recent Budget Group discussions (past 6 months) have focused on:

Recurring Topics:
1. Collections Budget Allocation
   - Ongoing debate about digital vs print resources
   - 20% shift toward electronic proposed (2024-01 meeting)

2. Personnel Costs
   - Salary increase requests reviewed monthly
   - Position freeze evaluation underway

3. Grant Funding
   - New grant opportunities discussed in March meeting
   - Proposal deadlines upcoming in April

Recent Action Items:
- Review FY2025 budget proposals (assigned to Committee members)
- Prepare cost-benefit analysis for digital subscriptions
- Coordinate with Development on fundraising targets

Key Participants:
- Bill Maher (Committee Chair)
- Joanne Kaczmarek (frequent contributor)
- Finance staff presentations

Last meeting: 2024-03-15 - Review those minutes for most recent context.
```

---

## 8. Documentation & Deliverables

### Documentation to Create

**1. README Section** (Add to existing README.md)
```markdown
## GraphRAG Integration

### Quick Start

Install Ollama:
```bash
curl https://ollama.ai/install.sh | sh
ollama pull nomic-embed-text
ollama pull llama3.2
```

Run embedding generation (one-time setup):
```python
# In is547_project.ipynb, execute GraphRAG cells 1-3
```

Ask questions:
```python
from data_pipeline.graphrag_qa import ask_question
response = ask_question(rag, "What were the major 2023 initiatives?")
```

### Architecture

- **Embeddings:** nomic-embed-text (768d) via Ollama
- **Vector Index:** Neo4j native vector index (cosine similarity)
- **Retrieval:** HybridCypherRetriever (vector + fulltext + graph)
- **LLM:** llama3.2 via Ollama
- **Storage:** Embeddings stored as Document.embedding property

### Example Queries

See `data_pipeline/graphrag_qa.py:example_questions()` for inspiration.
```

**2. Module Docstrings**
- Each new Python file needs comprehensive docstrings
- Explain purpose, parameters, return values
- Include usage examples

**3. Notebook Markdown Cells**
- Explain what each cell does
- Provide expected output examples
- Include troubleshooting tips

**4. Configuration Guide**
- Document all .env variables
- Explain model selection trade-offs
- Provide hardware recommendations

### Code Quality Standards

**Follow Existing Patterns:**
1. Type hints for function parameters
2. Descriptive variable names
3. Error handling with try/except
4. Progress reporting for long operations
5. Docstrings for all public functions

**Example Function Template:**
```python
def example_function(
    param1: str,
    param2: int = 10,
    optional_param: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Brief description of what the function does.

    Args:
        param1: Description of param1
        param2: Description with default value (default: 10)
        optional_param: Optional parameter description

    Returns:
        Dictionary containing result data with keys:
        - 'key1': Description of key1
        - 'key2': Description of key2

    Raises:
        ValueError: When param1 is invalid
        ConnectionError: When Neo4j connection fails

    Example:
        >>> result = example_function("test", param2=20)
        >>> print(result['key1'])
        'expected output'
    """
    # Implementation
    pass
```

### Testing Checklist

Before considering feature complete:

- [ ] All dependencies install without errors
- [ ] Ollama connection verified
- [ ] All 2,198 documents have embeddings
- [ ] Vector index is ONLINE and queryable
- [ ] Fulltext index is ONLINE
- [ ] Vector retriever returns results
- [ ] Hybrid retriever returns better results than vector alone
- [ ] LLM generates coherent responses
- [ ] Example questions produce expected answers
- [ ] Filtered queries work (by committee, date)
- [ ] Error handling gracefully skips problematic documents
- [ ] Progress reporting works during long operations
- [ ] Documentation is clear and complete
- [ ] Code follows existing project patterns

---

## 9. Implementation Timeline

### Estimated Effort

**Total Time: 8-12 hours** (assuming familiarity with Python and Neo4j)

| Phase | Task | Time Estimate |
|-------|------|---------------|
| 1 | Install dependencies, configure .env | 30 minutes |
| 2 | Implement generate_embeddings.py | 2 hours |
| 3 | Implement setup_vector_index.py | 1 hour |
| 4 | Implement graphrag_retriever.py | 1.5 hours |
| 5 | Implement graphrag_qa.py | 1.5 hours |
| 6 | Create notebook cells | 1 hour |
| 7 | Implement graphrag_verification.py | 1 hour |
| 8 | Testing and debugging | 2 hours |
| 9 | Documentation | 1.5 hours |

**Runtime (One-Time Setup):**
- Embedding generation: ~60-90 minutes (2,198 documents)
- Vector index creation: ~5 minutes
- Testing: ~15 minutes

### Recommended Implementation Order

**Day 1: Core Infrastructure**
1. Update requirements.txt and .env
2. Implement generate_embeddings.py
3. Implement setup_vector_index.py
4. Generate embeddings for 100 test documents
5. Create vector index and verify it works

**Day 2: Retrieval & RAG**
6. Implement graphrag_retriever.py
7. Test vector search with sample queries
8. Implement graphrag_qa.py
9. Test Q&A with sample questions

**Day 3: Integration & Polish**
10. Add notebook cells
11. Run full embedding generation (2,198 docs)
12. Implement graphrag_verification.py
13. Run comprehensive tests
14. Write documentation

---

## 10. Success Metrics

### Quantitative Metrics

**Coverage:**
- Target: 100% of documents have embeddings (2,198/2,198)
- Acceptable: ≥95% (may skip corrupted files)

**Performance:**
- Retrieval time: <1 second for vector search
- RAG response time: <15 seconds end-to-end
- Embedding generation: <2 hours for full dataset

**Quality:**
- Retrieval relevance: Top-5 results should include ≥2 truly relevant docs
- Answer accuracy: LLM should cite correct source documents
- Zero crashes or unhandled exceptions during normal operation

### Qualitative Metrics

**Usability:**
- Non-technical users can ask natural language questions
- Error messages are clear and actionable
- Documentation is sufficient for reproduction

**Usefulness:**
- Answers questions that couldn't be answered with keyword search alone
- Provides committee context automatically
- Surfaces connections between documents via graph traversal

**Maintainability:**
- Code follows existing project patterns
- New documents can be added incrementally
- Updates to embeddings can be performed selectively

---

## 11. References & Resources

### Official Documentation

- [neo4j-graphrag-python User Guide](https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_rag.html)
- [Neo4j Vector Indexes Documentation](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/)
- [Ollama Embeddings Documentation](https://ollama.com/blog/embedding-models)
- [Ollama Models Library](https://ollama.com/search?c=embedding)

### Tutorials & Blog Posts

- [Hybrid Retrieval for GraphRAG Applications](https://neo4j.com/blog/developer/hybrid-retrieval-graphrag-python-package/)
- [Running GraphRAG Locally with Neo4j and Ollama](https://sandeep14.medium.com/running-graphrag-locally-with-neo4j-and-ollama-text-format-371bf88b14b7)
- [Neo4j GraphRAG Python Package Guide](https://neo4j.com/developer/genai-ecosystem/graphrag-python/)

### GitHub Examples

- [neo4j-graphrag-python Examples](https://github.com/neo4j/neo4j-graphrag-python/tree/main/examples)
- [HybridCypherRetriever Example](https://github.com/neo4j/neo4j-graphrag-python/blob/main/examples/retrieve/hybrid_cypher_retriever.py)
- [GraphRAG-Ollama-Neo4J Project](https://github.com/StevenBtw/GraphRAG-Ollama-Neo4J)

### Research Papers (Background)

- nomic-embed-text paper: [Nomic Embed: Training a Reproducible Long Context Text Embedder](https://arxiv.org/abs/2402.01613)
- GraphRAG concept: [From Local to Global: A Graph RAG Approach](https://arxiv.org/abs/2404.16130)

---

## 12. Appendix: File Structure After Implementation

```
IS547_Project/
├── .env                                      # Updated with GraphRAG config
├── requirements.txt                          # Updated with neo4j-graphrag
├── is547_project.ipynb                       # Updated with GraphRAG cells
├── README.md                                 # Updated with GraphRAG section
├── data_pipeline/
│   ├── neo4j_import.py                       # Existing (unchanged)
│   ├── nlp_term_extraction_preview.py        # Existing (reused)
│   ├── generate_embeddings.py                # NEW: Embedding generation
│   ├── setup_vector_index.py                 # NEW: Index management
│   ├── graphrag_retriever.py                 # NEW: Retrieval logic
│   ├── graphrag_qa.py                        # NEW: Q&A interface
│   └── graphrag_verification.py              # NEW: Testing utilities
└── docs/
    └── features/
        └── feature-graphrag-ollama-integration.md  # This document
```

**Unchanged Files:**
- All existing data_pipeline modules remain functional
- No modifications to data/ directory structure
- No changes to existing notebook cells (new cells appended)

**New Dependencies:**
- neo4j-graphrag (with ollama extra)
- ollama (Python client)

**Neo4j Database Changes:**
- Document nodes gain `embedding` property (LIST<FLOAT>)
- Two new indexes: `document_embeddings` (vector), `document_fulltext` (fulltext)
- No changes to existing nodes, relationships, or properties

---

## Conclusion

This implementation plan provides a comprehensive roadmap for adding GraphRAG capabilities to the IS547 committee documents project. The feature leverages the official `neo4j-graphrag-python` library with local Ollama models to enable privacy-preserving semantic search and question-answering.

**Key Advantages:**
- ✅ Builds on existing infrastructure (Neo4j graph, text extraction)
- ✅ No external API calls or costs
- ✅ Incremental implementation (can be built in phases)
- ✅ Follows existing code patterns
- ✅ Comprehensive testing and verification steps

**Next Steps:**
1. Review this plan with stakeholders
2. Install Ollama and test models
3. Begin implementation with Phase 1 (dependencies)
4. Implement and test each module incrementally
5. Run full embedding generation
6. Document learnings and edge cases

The estimated 8-12 hour implementation time does not include the ~90 minute one-time embedding generation runtime. Testing with a subset of documents (e.g., 100 files) is recommended before processing the full dataset.

---

**Document Version:** 1.0
**Last Updated:** 2026-01-11
**Feature Status:** Ready for Implementation
**Plan Saved to:** `docs/features/feature-graphrag-ollama-integration.md`

---

## Sources

- [User Guide: RAG — neo4j-graphrag-python documentation](https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_rag.html)
- [neo4j_graphrag.embeddings.ollama — neo4j-graphrag-python documentation](https://neo4j.com/docs/neo4j-graphrag-python/current/_modules/neo4j_graphrag/embeddings/ollama.html)
- [neo4j_graphrag.llm.ollama_llm — neo4j-graphrag-python documentation](https://neo4j.com/docs/neo4j-graphrag-python/current/_modules/neo4j_graphrag/llm/ollama_llm.html)
- [Hybrid Retrieval for GraphRAG Applications Using the GraphRAG Python Package](https://neo4j.com/blog/developer/hybrid-retrieval-graphrag-python-package/)
- [neo4j-graphrag-python GitHub Repository](https://github.com/neo4j/neo4j-graphrag-python)
- [Vector indexes - Cypher Manual](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/)
- [Vector Indexes | GraphAcademy](https://graphacademy.neo4j.com/courses/llm-fundamentals/2-vectors-semantic-search/2-vector-index/)
- [nomic-embed-text - Ollama](https://ollama.com/library/nomic-embed-text)
- [Embedding models · Ollama Blog](https://ollama.com/blog/embedding-models)
- [Running GraphRAG Locally with Neo4j and Ollama](https://sandeep14.medium.com/running-graphrag-locally-with-neo4j-and-ollama-text-format-371bf88b14b7)
