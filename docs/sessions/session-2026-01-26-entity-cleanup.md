# Work Session: 2026-01-26 - Entity Cleanup for Graph Dataset

## Goals
- Create filtered minutes-only dataset for GraphRAG processing
- Clean up NLP entity quality (PERSON, ORG, GPE) with stricter validation
- Prepare data for knowledge graph and semantic search

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

## Next Steps

1. ~~Update notebook with cells to run filtering and cleanup~~ **DONE**
2. Proceed with GraphRAG implementation (per existing feature plan)
3. Consider regenerating knowledge graph visualization with cleaner entities

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

---

*Session documented: 2026-01-26*
*Duration: ~2 hours*
