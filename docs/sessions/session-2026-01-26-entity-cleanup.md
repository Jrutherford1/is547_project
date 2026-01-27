# Work Session: 2026-01-26 - Entity Cleanup & GraphRAG Implementation

## Goals
- Create filtered minutes-only dataset for GraphRAG processing
- Clean up NLP entity quality (PERSON, ORG, GPE) with stricter validation
- Prepare data for knowledge graph and semantic search
- Implement full GraphRAG integration with Ollama for semantic Q&A

## What Happened

### Phase 1: Create Filtered Minutes Dataset

**Created new module:** `data_pipeline/filter_for_graph.py`

**Purpose:** Extract only Minutes documents into a flattened structure for graph processing.

**Functions:**
- `scan_minutes_documents(source_dir)` - Scans for Minutes by checking JSON metadata `additionalType == "Minutes"`
- `copy_minutes_to_graph_folder(source_dir, dest_dir)` - Copies to flattened structure
- `filter_for_graph()` - Main entry point

**Results:**
- Source: `data/Processed_Committees/` (2,198 documents, 81 committees)
- Output: `data/committees_processed_for_graph/`
- 1,143 Minutes documents from 26 committees
- 55 committees skipped (no minutes)
- Flattened structure: `[Committee Name]/[files]` (no Minutes subfolder)

### Phase 2: Entity Quality Analysis

**Initial entity sample showed problems:**

| Entity Type | Issue | Examples |
|-------------|-------|----------|
| PERSON | First names only | Chris, Victor, Jen, Joe |
| PERSON | Non-persons | Gateway |
| ORG | Acronyms | EC, DEIA, TF, AP, CAPT |
| ORG | Misclassified persons | Kirstin Dougan, DoMonique |
| GPE | Misclassified persons | Merinda, Cindy, Atoma Batoma |
| GPE | Contraction artifacts | n't (Unicode U+2019) |

### Phase 3: Create Entity Cleanup Module

**Created new module:** `data_pipeline/cleanup_graph_entities.py`

**Purpose:** Reprocess entities with stricter validation for graph/RAG use. Kept separate from main pipeline for provenance.

**Key validation functions:**

1. **`is_valid_person_name()`**
   - Rejects single-word names (first names only)
   - Rejects all-caps short strings (acronyms)
   - Rejects strings starting with "the "
   - Uses expanded filter list (80+ terms)

2. **`is_valid_org()`**
   - Rejects short acronyms (2-5 chars, all caps)
   - Rejects things that look like person names (FirstName LastName pattern)
   - Rejects generic terms (committee, attendees, university, etc.)
   - Rejects software/system names (Primo, WordPress, LibGuides, etc.)
   - Rejects overly long phrases (>7 words)
   - Rejects meeting artifacts ("time and location", unbalanced parentheses)

3. **`is_valid_gpe()`**
   - Rejects common first names misclassified as locations
   - Rejects two-word patterns that look like person names
   - Rejects contraction artifacts (n't with U+2019 curly apostrophe)
   - Rejects entries with numbers
   - Rejects entries starting with "the "

**Filter term lists:**
- `ADDITIONAL_PERSON_FILTERS` - First names, non-persons, acronyms
- `ORG_FILTER_TERMS` - Acronyms, generic terms, software names
- `GPE_FILTER_TERMS` - Misclassified person names, non-locations
- `COMMON_FIRST_NAMES` - 150+ common first names for pattern detection

### Phase 4: Iterative Cleanup Runs

Ran cleanup multiple times, adding filters based on observed issues:

| Run | Issue Found | Filter Added |
|-----|-------------|--------------|
| 1 | First-name-only PERSON | `single_word_name` check |
| 2 | ORG acronyms (EC, AP) | `short_acronym` check |
| 3 | GPE person names (Merinda) | `COMMON_FIRST_NAMES` lookup |
| 4 | ORG too long phrases | `too_many_words` check |
| 5 | ORG unbalanced parens | `unbalanced_parens` check |
| 6 | GPE "n't" persisting | Unicode U+2019 explicit check |

**Unicode issue discovered:** The contraction "n't" uses RIGHT SINGLE QUOTATION MARK (U+2019), not ASCII apostrophe. Fixed with explicit `"\u2019"` in filter.

## Final Results

**Entity extraction statistics:**

| Entity Type | Extracted | Kept | Rejected | Rejection Rate |
|-------------|-----------|------|----------|----------------|
| PERSON | 31,094 | 13,656 | 17,438 | 56.1% |
| ORG | 22,429 | 8,183 | 14,246 | 63.5% |
| GPE | 1,942 | 909 | 1,033 | 53.2% |
| DATE | 9,592 | 9,575 | 17 | 0.2% |

**Unique entities in final dataset:**
- PERSON: 1,745 unique (all proper full names)
- ORG: 4,274 unique (real organizations)
- GPE: 398 unique (real locations)
- DATE: 3,974 unique

**Top PERSON entities (sample):**
1. Tom Teper (379 mentions)
2. John Wilkin (325)
3. Mary Laskowski (317)
4. Bill Mischo (257)
5. David Ward (256)

**Top ORG entities (sample):**
1. User Education Committee (102)
2. Administrative Council (94)
3. Technical Services (62)
4. Library Staff Support Committee (55)
5. iSchool (53)

**Top GPE entities (sample):**
1. Illinois (69)
2. Chicago (47)
3. Springfield (19)
4. India (18)
5. Michigan (16)

## Artifacts Produced

### New Modules
- `data_pipeline/filter_for_graph.py` (~150 lines)
- `data_pipeline/cleanup_graph_entities.py` (~600 lines)

### New Data
- `data/committees_processed_for_graph/` - 1,143 documents + 1,143 JSON files
- `data/graph_nlp_quality_report.json` - Quality metrics

### Key Design Decisions

1. **Separate cleanup module** - Kept `cleanup_graph_entities.py` separate from main `add_nlp_terms_to_metadata.py` for provenance tracking

2. **Aggressive filtering** - Prioritized removing garbage over keeping edge cases (56-63% rejection rates)

3. **Flattened directory structure** - Output uses `[Committee]/[files]` instead of `[Committee]/Minutes/[files]` since only Minutes are included

4. **Unicode-aware filtering** - Explicitly handle U+2019 (curly apostrophe) vs ASCII apostrophe

## Architectural Discussion

**Question raised:** Should entity extraction have been more thorough in the original JSON-LD? Will RAG make entities moot?

**Answer:** Entities and RAG serve complementary purposes:
- **Entities** → Graph structure (who, what, when), faceted search, relationship discovery
- **RAG embeddings** → Semantic understanding, conceptual search, Q&A

Neither makes the other obsolete. Entity quality matters for knowledge graph visualization; RAG captures full document semantics regardless of entity quality.

## Notebook Integration

Added 4 new cells to `is547_project.ipynb` after the Neo4j import section:

1. **Markdown:** "Graph Dataset Preparation for GraphRAG" - Section header explaining the purpose
2. **Code:** Filter to minutes-only dataset using `filter_for_graph()`
3. **Code:** Clean entities using `cleanup_graph_entities()`
4. **Code:** Review cleaned entities using `show_top_entities()`
5. **Markdown:** Summary of graph dataset and next steps

### Phase 5: GraphRAG Implementation

**Created 5 new modules:**

1. **`data_pipeline/generate_embeddings.py`** (~315 lines)
   - Generates vector embeddings using Ollama
   - Functions: `setup_embedder()`, `get_embedding()`, `extract_and_embed_documents()`, `check_embedding_status()`
   - Reads document text, generates 768-dim embeddings, stores in Neo4j

2. **`data_pipeline/setup_vector_index.py`** (~344 lines)
   - Creates Neo4j indexes for semantic search
   - Functions: `create_document_vector_index()`, `create_document_fulltext_index()`, `setup_all_indexes()`
   - Vector index: cosine similarity on Document.embedding
   - Fulltext index: keyword search on name, description, additionalType

3. **`data_pipeline/graphrag_retriever.py`** (~456 lines)
   - Semantic document retrieval
   - Search modes: `vector`, `hybrid`, `graph_enhanced`
   - Class: `GraphRAGRetriever` with `search()`, `find_by_person()`, `find_by_committee()`
   - Graph-enhanced returns documents with related people/orgs from graph

4. **`data_pipeline/graphrag_qa.py`** (~394 lines)
   - Q&A using retrieval + LLM generation
   - Functions: `ask_question()`, `create_qa_prompt()`, `format_context()`
   - Class: `GraphRAGQA` with `ask()` method
   - Includes interactive Q&A mode

5. **`data_pipeline/graphrag_verification.py`** (~496 lines)
   - End-to-end verification tests
   - Tests: Ollama connection, Neo4j connection, embedding coverage, indexes, vector search, LLM generation
   - Function: `run_full_verification()`

**Configuration updates:**

- **`requirements.txt`** - Added dependencies:
  - `neo4j~=5.27.0`
  - `neo4j-graphrag~=1.0.0`
  - `ollama~=0.4.0`
  - `python-dotenv~=1.0.0`

- **`.env`** - Added Ollama settings:
  ```
  OLLAMA_HOST=http://localhost:11434
  OLLAMA_EMBEDDING_MODEL=nomic-embed-text
  OLLAMA_LLM_MODEL=llama3.1:8b
  EMBEDDING_DIMENSIONS=768
  ```

**Bug fix:** Ollama API change - `ollama.list()` now returns objects with `.model` attribute instead of dict with `["name"]` key. Fixed in `graphrag_verification.py`.

**Verification results (10 test embeddings):**
- Ollama connection: ✓ (4 models available)
- Neo4j connection: ✓ (13,659 nodes)
- Vector index: ✓ ONLINE
- Fulltext index: ✓ ONLINE
- Embedding generation: ✓ (768 dims, ~150ms)
- Vector search: ✓ (returns ranked results)
- LLM generation: ✓ (~3-5 sec response)
- Q&A: ✓ (generates contextual answers with sources)

**Notebook cells added (6 new):**
1. Setup and verify Ollama/Neo4j connections
2. Create vector and fulltext indexes
3. Generate embeddings for all documents
4. Test semantic search
5. Test Q&A functionality
6. Run full verification suite

### Phase 6: Neo4j Re-import with Clean Data

**Problem:** Original Neo4j import used `data/Processed_Committees/` (dirty entities). Needed to re-import from cleaned dataset.

**Solution:** Re-ran `import_to_neo4j()` with `clear_first=True` pointing to `data/committees_processed_for_graph/`.

**New Neo4j Statistics:**
```
documents: 1143
committees: 26
persons: 725 (full names, ≥2 mentions)
organizations: 4274
locations: 398
relationships: 44547
```

**Added notebook cell:** Step 4 in Graph Dataset section for re-importing cleaned data.

### Phase 7: Full Embedding Generation

**Ran embeddings on cleaned dataset:**
- Documents processed: 1,143
- Embeddings generated: ~1,058
- Skipped: ~85 (legacy .doc/.ppt formats not supported)

**Embedding coverage:** ~93% of Minutes documents have embeddings.

**Legacy format errors:** `.doc` and `.ppt` files can't be processed (need additional libraries). These are mostly Related Documents that were filtered out anyway.

### Phase 8: Bug Fix - Retriever Property Warning

**Issue:** Neo4j warning `property 'committee' does not exist` when running searches.

**Cause:** Retriever queries referenced `node.committee` as a property, but committee is stored via `BELONGS_TO` relationship, not as a Document property.

**Fix:** Updated `graphrag_retriever.py` in three places:
- `vector_search()` - Added `OPTIONAL MATCH (node)-[:BELONGS_TO]->(c:Committee)` and changed to `c.name AS committee`
- `hybrid_search()` - Same fix
- `graph_enhanced_search()` - Removed redundant `committeeName`, use `c.name AS committee`

### Phase 9: Final Notebook Cells

**Added 2 new cells for user interaction:**

1. **Quick Q&A Cell** - Single question with answer and sources
   ```python
   question = "What budget discussions took place?"
   with GraphRAGQA() as qa:
       result = qa.ask(question, top_k=5)
   ```

2. **Interactive Mode Cell** - Continuous Q&A session
   ```python
   from data_pipeline.graphrag_qa import interactive_qa
   interactive_qa()  # Type 'quit' to exit
   ```

## Final System State

**Neo4j Database:**
- 1,143 Minutes documents with clean entities
- 725 persons (full names only)
- 4,274 organizations
- 398 locations
- Vector embeddings on ~93% of documents
- Vector index `document_embeddings` ONLINE
- Fulltext index `document_fulltext` ONLINE

**Capabilities:**
- Browse graph in Neo4j Browser (http://localhost:7474)
- Semantic search via vector similarity
- Hybrid search (vectors + keywords + graph)
- Natural language Q&A with source citations
- Interactive Q&A mode

## Next Steps

1. ~~Update notebook with cells to run filtering and cleanup~~ **DONE**
2. ~~Implement GraphRAG with Ollama~~ **DONE**
3. ~~Add notebook cells for GraphRAG~~ **DONE**
4. ~~Generate embeddings for cleaned dataset~~ **DONE**
5. ~~Re-import Neo4j with clean data~~ **DONE**
6. ~~Fix retriever property warning~~ **DONE**
7. ~~Add Q&A interaction cells~~ **DONE**
8. Consider regenerating knowledge graph visualization with cleaner entities
9. Consider adding support for legacy .doc/.ppt formats

## Usage

```python
# Filter minutes to graph dataset
from data_pipeline.filter_for_graph import filter_for_graph
result = filter_for_graph()

# Clean up entities
from data_pipeline.cleanup_graph_entities import cleanup_graph_entities, show_top_entities
result = cleanup_graph_entities()
show_top_entities(top_n=20)
```

## GraphRAG Usage Examples

```python
# Quick Q&A
from data_pipeline.graphrag_qa import GraphRAGQA

with GraphRAGQA() as qa:
    result = qa.ask("What budget discussions took place?")
    print(result['answer'])
    for s in result['sources']:
        print(f"  - {s['name']}")

# Interactive mode
from data_pipeline.graphrag_qa import interactive_qa
interactive_qa()  # Type 'quit' to exit, 'examples' for sample questions

# Search only (no LLM)
from data_pipeline.graphrag_retriever import GraphRAGRetriever

with GraphRAGRetriever() as r:
    docs = r.search("diversity initiatives", top_k=5)
    for d in docs:
        print(f"{d['name']} - {d.get('score', 0):.3f}")

# Find documents by person
with GraphRAGRetriever() as r:
    docs = r.find_by_person("Tom Teper", top_k=10)

# Find documents by committee
with GraphRAGRetriever() as r:
    docs = r.find_by_committee("Executive Committee", top_k=20)
```

---

*Session documented: 2026-01-26*
*Duration: ~4 hours*
