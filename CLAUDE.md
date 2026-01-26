# IS547_Project - Library Committee Documents Archive

## Overview

Data curation and archival project managing ~2,200 institutional library committee documents migrated from a legacy WordPress site. The project enhances access to institutional memory through:

- Consistent file naming conventions with date extraction
- NLP-extracted metadata (PERSON, ORG, GPE, DATE entities) using spaCy
- SHA-256 checksums for fixity verification
- Schema.org JSON-LD metadata for each document
- Interactive knowledge graph visualization

## Important Warnings

- **DO NOT explore `data/Committees/` or `data/Processed_Committees/`** - contains thousands of files
- Processing the full dataset requires significant time due to NLP extraction
- The `knowledge_graph_explorer.html` contains embedded data (~85KB)

## Directory Structure

```
IS547_Project/
├── data/
│   ├── Committees/              # ~2,203 source documents (AVOID)
│   ├── Processed_Committees/    # Output with JSON metadata (AVOID)
│   ├── names.csv                # Initial filename extraction
│   ├── manually_updated_committee_names.csv
│   └── final_updated_committee_names.csv
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
│   └── project_metadata.py      # Project-level metadata generation
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

## Key Dependencies

```
spacy              # NLP (requires: python -m spacy download en_core_web_sm)
pandas             # Data manipulation
networkx           # Graph construction
pyvis              # Interactive graph visualization
pdfplumber         # PDF text extraction
python-docx        # Word document parsing
python-pptx        # PowerPoint parsing
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

## Knowledge Graph Features

The `knowledge_graph_explorer.html` visualization provides:

- **Document nodes** (red) - Meeting minutes, agendas, related documents
- **Person nodes** (blue) - Individuals extracted via NLP, sized by mention count
- **Click filtering** - Click a person to show only their document connections
- **Search** - Filter by name or document type
- **Hover tooltips** - Detailed node information
- **Freeze/unfreeze** - Toggle physics simulation

## Architecture Notes

- **Two-pass processing**: NLP and graph building count entities first, then build
- **Batch processing**: spaCy uses `pipe()` with batch_size=50 for performance
- **Entity filtering**: Generic terms (Library, Staff, Committee, etc.) filtered from PERSON entities
- **Static layout**: Knowledge graph uses frozen physics for stable viewing
- **No server required**: Visualization runs entirely client-side
- **Neo4j export**: Supports both Cypher scripts and CSV bulk import formats
- **Quality validation**: Text quality scoring and entity validation filter garbage/corrupted extractions
- **Quality reporting**: Track entity rejection rates and identify problematic source documents

## Planned Features

### GraphRAG Ollama Integration (Planned)
Adds local, privacy-preserving semantic search and Q&A capabilities using Neo4j GraphRAG + Ollama.

**Status:** Fully planned, not yet implemented
**Plan location:** `docs/features/feature-graphrag-ollama-integration.md`

**Capabilities:**
- Semantic document search via vector embeddings
- Hybrid retrieval (semantic + keyword + graph)
- Natural language Q&A about committees, people, topics
- 100% local processing (no external APIs)

**Models:**
- Embeddings: nomic-embed-text (768-dim)
- LLM: llama3.2 (8B parameters, 128K context)

**Implementation estimate:** 8-12 hours coding + 90 minutes runtime
