# IS547_Project - Library Committee Documents Archive

## Overview

Data curation and archival project managing ~2,200 institutional library committee documents migrated from a legacy WordPress site. The project enhances access to institutional memory through:

- Consistent file naming conventions with date extraction
- NLP-extracted metadata (PERSON, ORG, GPE, DATE entities) using spaCy
- SHA-256 checksums for fixity verification
- Schema.org JSON-LD metadata for each document
- Interactive knowledge graph visualization
- GraphRAG semantic search and Q&A using Neo4j + Ollama (100% local)

## Important Warnings

- **DO NOT explore `data/Committees/` or `data/Processed_Committees/`** - contains thousands of files
- Processing the full dataset requires significant time due to NLP extraction
- The `knowledge_graph_explorer.html` contains embedded data (~85KB)
- **GraphRAG requires Neo4j running locally** - ensure Neo4j is installed and .env configured
- Embedding generation takes ~90 minutes for the full cleaned dataset

## Directory Structure

```
IS547_Project/
├── data/
│   ├── Committees/              # ~2,203 source documents (AVOID)
│   ├── Processed_Committees/    # Output with JSON metadata (AVOID)
│   ├── committees_processed_for_graph/  # Cleaned dataset for Neo4j import
│   ├── names.csv                # Initial filename extraction
│   ├── manually_updated_committee_names.csv
│   ├── final_updated_committee_names.csv
│   ├── nlp_quality_report.json  # Quality metrics for full dataset
│   └── graph_nlp_quality_report.json  # Quality metrics for graph dataset
├── data_pipeline/               # Core Python processing modules
│   ├── data_explore.py          # File discovery utilities
│   ├── data_cleaning.py         # Copy files, remove .DS_Store
│   ├── file_naming.py           # Date extraction, name generation
│   ├── final_file_naming.py     # Final renaming with collision handling
│   ├── enhance_metadata.py      # JSON-LD metadata + SHA-256 checksums
│   ├── add_nlp_terms_to_metadata.py  # Batch NLP entity extraction
│   ├── text_quality.py          # Text quality validation (pre-NLP)
│   ├── entity_validation.py     # Entity filtering (post-NLP)
│   ├── nlp_quality_report.py    # Quality tracking and reporting
│   ├── nlp_term_extraction_preview.py # Text extraction from docs
│   ├── build_redacted_knowlege_graph.py # Knowledge graph builder
│   ├── neo4j_export.py          # Export to Neo4j (Cypher + CSV)
│   ├── metadata_check.py        # Validation utilities
│   ├── project_metadata.py      # Project-level metadata generation
│   ├── generate_embeddings.py   # Generate & store vector embeddings in Neo4j
│   ├── setup_vector_index.py    # Create Neo4j vector & fulltext indexes
│   ├── graphrag_retriever.py    # Semantic & hybrid search retrievers
│   ├── graphrag_qa.py           # Natural language Q&A with Ollama LLM
│   └── graphrag_verification.py # Test GraphRAG pipeline end-to-end
├── lib/                         # Frontend visualization libraries
│   ├── vis-9.1.2/              # vis.js network visualization
│   ├── tom-select/             # Dropdown filter library
│   └── bindings/utils.js       # Custom JS utilities
├── is547_project.ipynb          # Main orchestration notebook
├── knowledge_graph_explorer.html # Interactive person-document graph
├── project_metadata.jsonld      # Dataset-level Schema.org metadata
├── docs/                        # Project documentation
│   ├── sessions/                # Work session logs
│   └── features/                # Feature implementation plans
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## Data Pipeline Workflow

The pipeline processes documents through these stages:

1. **Explore** (`data_explore.py`) - Scan source directories, count files, identify types
2. **Clean** (`data_cleaning.py`) - Copy to Processed_Committees, remove .DS_Store
3. **Name** (`file_naming.py`) - Extract dates from filenames using regex patterns
4. **Finalize Names** (`final_file_naming.py`) - Apply manual corrections, handle collisions
5. **Enhance Metadata** (`enhance_metadata.py`) - Create JSON-LD files with SHA-256 checksums
6. **NLP Extraction** (`add_nlp_terms_to_metadata.py`) - Extract entities with spaCy batch processing
   - Includes quality validation via `text_quality.py` and `entity_validation.py`
   - Generates quality reports via `nlp_quality_report.py`
7. **Build Graph** (`build_redacted_knowlege_graph.py`) - Generate interactive knowledge graph
8. **Neo4j Export** (`neo4j_export.py`) - Export to Neo4j graph database (optional)
9. **GraphRAG Setup** (GraphRAG modules) - Enable semantic search and Q&A
   - Generate embeddings with `generate_embeddings.py`
   - Create indexes with `setup_vector_index.py`
   - Query with `graphrag_retriever.py` and `graphrag_qa.py`

## GraphRAG Workflow

The GraphRAG implementation adds semantic search and Q&A on top of the Neo4j knowledge graph:

1. **Prerequisites**
   - Neo4j database running locally (or remote)
   - Ollama installed with models pulled (`nomic-embed-text`, `llama3.1:8b`)
   - `.env` file configured with connection details

2. **Data Preparation**
   - Cleaned dataset in `committees_processed_for_graph/`
   - Neo4j import completed via `neo4j_export.py`
   - Graph contains Document, Person, Organization, Location nodes

3. **Embedding Generation** (`generate_embeddings.py`)
   - Extracts text from each document
   - Generates 768-dim vector using Ollama nomic-embed-text
   - Stores embedding as property on Document node in Neo4j
   - Progress tracking with batch processing

4. **Index Creation** (`setup_vector_index.py`)
   - Creates vector index on Document.embedding for similarity search
   - Creates fulltext index on Document text properties for keyword matching
   - Verifies index population and readiness

5. **Retrieval** (`graphrag_retriever.py`)
   - **Vector search**: Pure semantic similarity using cosine distance
   - **Hybrid search**: Combines vector + fulltext with configurable weights
   - **Graph-enhanced**: Includes related committee/person context from graph traversal

6. **Q&A** (`graphrag_qa.py`)
   - Retrieves relevant documents using chosen retrieval mode
   - Constructs prompt with document context
   - Generates answer using Ollama LLM
   - Cites source documents in response

7. **Verification** (`graphrag_verification.py`)
   - End-to-end testing of the pipeline
   - Validates embedding presence
   - Tests all retrieval modes
   - Confirms Q&A functionality

## Key Dependencies

```
spacy              # NLP (requires: python -m spacy download en_core_web_sm)
pandas             # Data manipulation
networkx           # Graph construction
pyvis              # Interactive graph visualization
pdfplumber         # PDF text extraction
python-docx        # Word document parsing
python-pptx        # PowerPoint parsing

# GraphRAG dependencies
neo4j              # Graph database driver
neo4j-graphrag     # Neo4j GraphRAG toolkit
ollama             # Local LLM and embeddings
python-dotenv      # Environment variable management
```

## Common Tasks

### Run the full pipeline
Execute cells sequentially in `is547_project.ipynb`

### Regenerate knowledge graph only
```python
from data_pipeline.build_redacted_knowlege_graph import create_person_document_explorer
create_person_document_explorer()
```

### Check entity extraction
```python
from data_pipeline.metadata_check import check_person_entities
check_person_entities()
```

### Preview NLP extraction on sample files
```python
from data_pipeline.nlp_term_extraction_preview import run_entity_preview
run_entity_preview(limit=50)
```

### Reprocess entities with improved validation
```python
from data_pipeline.add_nlp_terms_to_metadata import reprocess_all_entities
reprocess_all_entities(report_path="data/nlp_quality_report.json")
```
This clears existing entities and re-extracts with quality filtering

### Export to Neo4j
```python
from data_pipeline.neo4j_export import export_to_neo4j
export_to_neo4j(output_format="both")  # Creates Cypher + CSV
```
Output: `data/neo4j_export/neo4j_import.cypher` and `data/neo4j_export/csv/`

### Setup GraphRAG (requires Neo4j running)

1. **Configure environment** - Create `.env` file:
```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
OLLAMA_HOST=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_LLM_MODEL=llama3.1:8b
EMBEDDING_DIMENSIONS=768
```

2. **Generate embeddings** (takes ~90 minutes):
```python
from data_pipeline.generate_embeddings import extract_and_embed_documents, check_embedding_status
extract_and_embed_documents(base_dir='data/committees_processed_for_graph')
check_embedding_status()
```

3. **Create vector indexes**:
```python
from data_pipeline.setup_vector_index import setup_all_indexes
setup_all_indexes()
```

4. **Test the pipeline**:
```python
from data_pipeline.graphrag_verification import run_full_verification
run_full_verification()
```

### Use GraphRAG for Semantic Search

```python
from data_pipeline.graphrag_retriever import GraphRAGRetriever

retriever = GraphRAGRetriever()

# Vector search
results = retriever.vector_search(
    query="diversity initiatives",
    top_k=5
)

# Hybrid search (vector + keyword)
results = retriever.hybrid_search(
    query="budget planning",
    top_k=5,
    vector_weight=0.7
)

# Graph-enhanced search
results = retriever.graph_enhanced_search(
    query="faculty hiring policies",
    top_k=5,
    include_context=True
)
```

### Use GraphRAG for Q&A

```python
from data_pipeline.graphrag_qa import ask_question

# Simple Q&A
response = ask_question(
    question="What are the main diversity initiatives discussed?",
    search_type="hybrid",
    top_k=5
)
print(response)

# Q&A with full context
response = ask_question(
    question="Who were the key people involved in budget planning?",
    search_type="graph_enhanced",
    top_k=3
)
```

## Metadata Format

Each document has a companion `.json` file with Schema.org JSON-LD:

```json
{
  "@context": {"@vocab": "http://schema.org/"},
  "@type": "CreativeWork",
  "name": "Committee_Minutes_2023-05-15.pdf",
  "dateCreated": "2023-05-15",
  "fileFormat": "PDF",
  "additionalType": "Minutes",
  "creator": {"@type": "Organization", "name": "Library Staff"},
  "entities": {
    "PERSON": ["John Smith", "Jane Doe"],
    "ORG": ["Library Department"],
    "GPE": ["Illinois"],
    "DATE": ["2023-05-15"]
  },
  "keywords": ["John Smith", "Library Department"],
  "checksum": {
    "algorithm": "SHA-256",
    "value": "abc123..."
  },
  "license": "Open/Public per institutional policy"
}
```

## File Types Processed

- `.docx` (1,764 files) - Word documents
- `.pdf` (333 files) - PDF documents
- `.doc` (53 files) - Legacy Word
- `.ppt/.pptx` (47 files) - PowerPoint
- `.xls/.xlsx` (6 files) - Excel

## Cleaned Graph Dataset

The `data/committees_processed_for_graph/` directory contains a curated subset of documents optimized for Neo4j import and GraphRAG:

**Quality criteria:**
- Text quality score > threshold (filters corrupted/unreadable documents)
- Valid entity extraction (no garbage text, proper person/org/location names)
- Non-empty content (skips blank or metadata-only files)
- Proper date formatting

**Contents:**
- Document files (.pdf, .docx, etc.) meeting quality thresholds
- Companion `.json` metadata files with:
  - Extracted entities (PERSON, ORG, GPE, DATE)
  - Document properties (name, date, type, creator)
  - SHA-256 checksums for fixity
  - Quality metrics

**Purpose:**
- Provides clean dataset for Neo4j graph import
- Reduces noise in GraphRAG search results
- Improves LLM answer quality by removing low-quality sources
- Faster embedding generation (smaller corpus)

**Quality report:** `data/graph_nlp_quality_report.json` tracks:
- Total documents processed vs. accepted
- Entity rejection rates by type
- Common quality issues identified

## Knowledge Graph Features

### Interactive Visualization
The `knowledge_graph_explorer.html` visualization provides:

- **Document nodes** (red) - Meeting minutes, agendas, related documents
- **Person nodes** (blue) - Individuals extracted via NLP, sized by mention count
- **Click filtering** - Click a person to show only their document connections
- **Search** - Filter by name or document type
- **Hover tooltips** - Detailed node information
- **Freeze/unfreeze** - Toggle physics simulation

### GraphRAG Capabilities (IMPLEMENTED)
The Neo4j + Ollama integration provides:

- **Semantic search** - Find documents by meaning, not just keywords
- **Hybrid retrieval** - Combine vector similarity + fulltext matching
- **Graph-enhanced search** - Include committee/person context in results
- **Natural language Q&A** - Ask questions about documents in plain English
- **100% local processing** - No external APIs, complete privacy
- **Context-aware answers** - LLM generates responses grounded in actual documents

## Architecture Notes

- **Two-pass processing**: NLP and graph building count entities first, then build
- **Batch processing**: spaCy uses `pipe()` with batch_size=50 for performance
- **Entity filtering**: Generic terms (Library, Staff, Committee, etc.) filtered from PERSON entities
- **Static layout**: Knowledge graph uses frozen physics for stable viewing
- **No server required**: Visualization runs entirely client-side
- **Neo4j export**: Supports both Cypher scripts and CSV bulk import formats
- **Quality validation**: Text quality scoring and entity validation filter garbage/corrupted extractions
- **Quality reporting**: Track entity rejection rates and identify problematic source documents
- **Cleaned graph dataset**: `committees_processed_for_graph/` contains curated subset for Neo4j import
- **Vector embeddings**: 768-dimensional vectors generated by nomic-embed-text via Ollama
- **Hybrid search**: Combines cosine similarity (vector) + BM25 scoring (fulltext) with configurable weights
- **Graph context**: Search results can include related committee and person information from graph structure
- **LLM grounding**: Q&A responses cite source documents to prevent hallucination

## Implementation Status

### GraphRAG Ollama Integration (IMPLEMENTED)
Local, privacy-preserving semantic search and Q&A capabilities using Neo4j GraphRAG + Ollama.

**Status:** Fully implemented and operational
**Implementation date:** January 2026
**Documentation:** See "Use GraphRAG" sections above

**Implemented capabilities:**
- Semantic document search via vector embeddings
- Hybrid retrieval (semantic + keyword + graph)
- Natural language Q&A about committees, people, topics
- 100% local processing (no external APIs)
- Graph-enhanced context retrieval
- Quality-filtered dataset for optimal results

**Models in use:**
- Embeddings: nomic-embed-text (768-dimensional vectors)
- LLM: llama3.1:8b (8 billion parameters, 128K context window)

**Performance:**
- Embedding generation: ~90 minutes for cleaned dataset
- Query latency: <2 seconds for semantic search
- Q&A latency: 5-15 seconds depending on context size

**Dataset:**
- Graph dataset: `data/committees_processed_for_graph/` (curated subset)
- Quality report: `data/graph_nlp_quality_report.json`
- Neo4j nodes: Document, Person, Organization, Location nodes with relationships

## Future Enhancements

Potential additions to the project:

- **Advanced RAG techniques**: Implement query decomposition, multi-hop reasoning
- **Temporal analysis**: Time-series analysis of committee topics and participation
- **Network analysis**: Centrality metrics, community detection on person-document graph
- **Web interface**: Flask/FastAPI frontend for non-technical users
- **Batch Q&A**: Process multiple questions and generate reports
- **Fine-tuned models**: Train domain-specific embeddings on committee document corpus
