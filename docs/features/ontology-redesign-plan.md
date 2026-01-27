# Ontology and Data Structure Redesign Plan

## Executive Summary

The current implementation uses Schema.org's `CreativeWork` type as a generic container, treating committee documents as simple files with extracted entities rather than as semantically rich records of institutional activity. This plan proposes a multi-layered ontology approach that captures the true nature of committee documents: they are records of **Events** (meetings), involving **Agents** (people in roles), producing **Outcomes** (decisions, action items), with full **Provenance** tracking.

---

## 1. Assessment of Current Approach

### Current Data Model Analysis

**JSON-LD Structure (from `data/Processed_Committees/`)**

```json
{
    "@context": {"@vocab": "http://schema.org/"},
    "@type": "CreativeWork",
    "name": "Committee_Minutes_2022-05-23.docx",
    "creator": {"@type": "Organization", "name": "Library Staff"},
    "additionalType": "Minutes",
    "dateCreated": "2022-05-23",
    "entities": {
        "PERSON": ["Alex", "Chris Wiley", "Dan Tracy"],
        "ORG": ["Diversity Residency Advisory Committee"],
        "GPE": ["Atoma Batoma"],
        "DATE": ["July", "May 23, 2022"]
    }
}
```

### Strengths

| Aspect | Assessment |
|--------|------------|
| **Schema.org Foundation** | Good choice for interoperability and web discovery |
| **Checksum Integrity** | SHA-256 checksums provide solid fixity verification |
| **JSON-LD Format** | Enables linked data without complex infrastructure |
| **Entity Extraction** | Basic NER captures some semantic content |
| **Quality Validation** | Text quality scoring and entity filtering reduce garbage |

### Weaknesses

| Issue | Impact | Example |
|-------|--------|---------|
| **Flat Entity Lists** | No relationships captured | "Tom Teper" appears in document, but was he the author? Attendee? Subject of discussion? |
| **Missing Document Semantics** | Minutes are treated as generic "CreativeWork" | No distinction between: meeting occurred, decisions made, actions assigned |
| **No Role Modeling** | People appear without context | "Chair", "Secretary", "Guest" roles are lost |
| **No Event Modeling** | Meetings are implicit | The meeting that produced the minutes is not a first-class entity |
| **No Action/Decision Extraction** | Key outcomes not captured | "Motion passed", "Action: John to review budget" lost in text |
| **Limited Provenance** | Only file origin tracked | Who created the record? When was it modified? Who approved it? |
| **Generic Organization** | `creator: "Library Staff"` | Every document has identical creator - meaningless |
| **No Committee Relationships** | Committees exist only in folder paths | Parent/child committees, reporting structures not modeled |

### Current NLP Extraction Gaps

From `data_pipeline/add_nlp_terms_to_metadata.py`:

```python
entities = {
    "PERSON": [],  # Names only, no roles
    "ORG": [],     # Organizations mentioned, no relationship type
    "GPE": [],     # Locations, often misclassified
    "DATE": []     # Dates without temporal semantics
}
```

**Missing extractions:**
- Attendee vs. mentioned vs. absent distinctions
- Author/secretary/chair identification
- Motion/decision patterns ("Motion to approve...", "The committee decided...")
- Action item patterns ("ACTION:", "John will...", "by next meeting...")
- Agenda item structure
- Vote outcomes

---

## 2. Recommended Ontology Framework

### Multi-Vocabulary Approach

Rather than relying solely on Schema.org, we recommend a layered approach combining established vocabularies:

```
┌────────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                           │
│         Custom classes for committee-specific concepts         │
│   (MeetingMinutes, ActionItem, Decision, CommitteeRole, etc.) │
├────────────────────────────────────────────────────────────────┤
│                    DOMAIN LAYER                                │
│              W3C Organization Ontology (ORG)                   │
│    Organizations, Memberships, Roles, Posts, Sites             │
├────────────────────────────────────────────────────────────────┤
│                    PROVENANCE LAYER                            │
│                  PROV-O (W3C Provenance)                       │
│      Entity, Activity, Agent, Attribution, Derivation          │
├────────────────────────────────────────────────────────────────┤
│                    ARCHIVAL LAYER                              │
│            Records in Contexts (RiC-O) concepts                │
│     Record, RecordSet, RecordCreation, AuthorizedAgent         │
├────────────────────────────────────────────────────────────────┤
│                    BASE LAYER                                  │
│              Schema.org + Dublin Core Terms                    │
│    CreativeWork, Event, Person, Organization, dcterms          │
└────────────────────────────────────────────────────────────────┘
```

### Namespace Prefixes

```json
{
    "@context": {
        "@vocab": "http://schema.org/",
        "org": "http://www.w3.org/ns/org#",
        "prov": "http://www.w3.org/ns/prov#",
        "dcterms": "http://purl.org/dc/terms/",
        "foaf": "http://xmlns.com/foaf/0.1/",
        "ric": "https://www.ica.org/standards/RiC/ontology#",
        "lib": "http://library.example.org/ontology#"
    }
}
```

### Core Classes

#### 1. Meeting Event (Central Entity)

```turtle
lib:Meeting a owl:Class ;
    rdfs:subClassOf schema:Event, prov:Activity ;
    rdfs:label "Committee Meeting" ;
    rdfs:comment "A scheduled gathering of committee members" .

# Properties
lib:hasAgenda           -> lib:AgendaItem (ordered list)
lib:hasAttendee         -> lib:MeetingParticipation
lib:producesMinutes     -> lib:MeetingMinutes
lib:producesDecision    -> lib:Decision
lib:producesActionItem  -> lib:ActionItem
lib:followsMeeting      -> lib:Meeting (previous meeting)
lib:hasQuorum           -> xsd:boolean
lib:scheduledDuration   -> xsd:duration
lib:actualDuration      -> xsd:duration
```

#### 2. Meeting Participation (Role-Based Attendance)

```turtle
lib:MeetingParticipation a owl:Class ;
    rdfs:subClassOf prov:Association ;
    rdfs:comment "Links a person to a meeting with their role" .

lib:participationRole   -> lib:MeetingRole
lib:attendanceStatus    -> lib:AttendanceStatus  # Present, Absent, Excused
lib:participant         -> foaf:Person
lib:representedBy       -> foaf:Person  # For proxy attendance
```

#### 3. Committee (Organizational Unit)

```turtle
lib:Committee a owl:Class ;
    rdfs:subClassOf org:FormalOrganization ;
    rdfs:comment "A library committee or working group" .

lib:committeeType       -> lib:CommitteeType  # Standing, Ad-hoc, Task Force
lib:hasCharter          -> lib:CharterDocument
lib:reportingTo         -> lib:Committee  # Parent committee
lib:hasMeeting          -> lib:Meeting
lib:hasCurrentMember    -> lib:CommitteeMembership
lib:establishedDate     -> xsd:date
lib:dissolvedDate       -> xsd:date
```

#### 4. Meeting Minutes (Enhanced Document)

```turtle
lib:MeetingMinutes a owl:Class ;
    rdfs:subClassOf schema:CreativeWork, prov:Entity, ric:Record ;
    rdfs:comment "Official record of a committee meeting" .

# Existing properties retained
schema:name, schema:dateCreated, schema:fileFormat, schema:checksum

# New properties
lib:recordsEvent        -> lib:Meeting
lib:approvalStatus      -> lib:ApprovalStatus  # Draft, Approved, Amended
lib:approvedDate        -> xsd:date
lib:approvedBy          -> foaf:Person
lib:recordedBy          -> foaf:Person  # Secretary/minute-taker
lib:containsDecision    -> lib:Decision
lib:containsActionItem  -> lib:ActionItem
lib:previousVersion     -> lib:MeetingMinutes
prov:wasGeneratedBy     -> lib:Meeting
prov:wasAttributedTo    -> foaf:Person
```

#### 5. Decision (Outcomes)

```turtle
lib:Decision a owl:Class ;
    rdfs:subClassOf prov:Entity ;
    rdfs:comment "A decision made during a meeting" .

lib:decisionText        -> xsd:string
lib:decisionType        -> lib:DecisionType  # Motion, Consensus, Chair ruling
lib:voteOutcome         -> lib:VoteOutcome  # Passed, Failed, Tabled
lib:voteTally           -> lib:VoteTally  # For: N, Against: M, Abstain: K
lib:madeBy              -> lib:Meeting
lib:supersedes          -> lib:Decision
lib:implementedVia      -> lib:ActionItem
```

#### 6. Action Item (Tasks)

```turtle
lib:ActionItem a owl:Class ;
    rdfs:subClassOf prov:Entity ;
    rdfs:comment "A task assigned during a meeting" .

lib:actionDescription   -> xsd:string
lib:assignedTo          -> foaf:Person
lib:assignedDate        -> xsd:date
lib:dueDate             -> xsd:date
lib:status              -> lib:ActionStatus  # Open, InProgress, Completed, Cancelled
lib:completedDate       -> xsd:date
lib:resultingFrom       -> lib:Meeting
lib:referencedInMeeting -> lib:Meeting  # Follow-up mentions
lib:blockedBy           -> lib:ActionItem
```

### Enumeration Types

```turtle
lib:MeetingRole a owl:Class ;
    owl:oneOf (lib:Chair lib:ViceChair lib:Secretary lib:Member lib:Guest
               lib:ExOfficio lib:Recorder lib:Presenter lib:Observer) .

lib:AttendanceStatus a owl:Class ;
    owl:oneOf (lib:Present lib:Absent lib:Excused lib:Late lib:LeftEarly lib:Proxy) .

lib:ApprovalStatus a owl:Class ;
    owl:oneOf (lib:Draft lib:Circulated lib:Approved lib:Amended lib:Superseded) .

lib:ActionStatus a owl:Class ;
    owl:oneOf (lib:Open lib:InProgress lib:Completed lib:Deferred lib:Cancelled) .

lib:DecisionType a owl:Class ;
    owl:oneOf (lib:MotionPassed lib:MotionFailed lib:MotionTabled
               lib:ConsensusReached lib:ChairRuling lib:NoAction) .

lib:CommitteeType a owl:Class ;
    owl:oneOf (lib:Standing lib:AdHoc lib:TaskForce lib:WorkingGroup lib:Advisory) .
```

---

## 3. Recommended NLP Extraction Improvements

### Current vs. Proposed Pipeline

```
CURRENT:
Document → Text Extraction → spaCy NER → {PERSON, ORG, GPE, DATE} → JSON

PROPOSED:
Document → Text Extraction → Section Detection → Multiple NLP Passes:
    ├─ Role-aware NER (Chair:, Present:, Absent:)
    ├─ Dependency Parsing (subject-verb-object relations)
    ├─ Pattern Matching (motions, actions, decisions)
    ├─ Coreference Resolution (he/she → named person)
    ├─ Temporal Expression Normalization
    └─ Relation Extraction (who-did-what-to-whom)
→ Structured Meeting Model → JSON-LD
```

### Specific NLP Components

#### A. Section Detection

Minutes have predictable structure. Add section detection:

```python
SECTION_PATTERNS = {
    "attendance": r"(?i)(present|attendees|in attendance|members present)[:.\s]",
    "absent": r"(?i)(absent|excused|not present)[:.\s]",
    "agenda": r"(?i)(agenda|order of business)[:.\s]",
    "old_business": r"(?i)(old business|previous business|unfinished business)",
    "new_business": r"(?i)(new business|current business)",
    "action_items": r"(?i)(action items?|to.?do|tasks?|assignments?)[:.\s]",
    "adjournment": r"(?i)(meeting adjourned|adjourned at)"
}
```

#### B. Attendance Extraction with Roles

```python
# Pattern-based role detection
ATTENDANCE_PATTERNS = {
    "chair": r"(?i)chair(?:person)?[:\s]+([A-Z][a-z]+ [A-Z][a-z]+)",
    "secretary": r"(?i)secretary[:\s]+([A-Z][a-z]+ [A-Z][a-z]+)",
    "recorder": r"(?i)(?:recorded|minutes) by[:\s]+([A-Z][a-z]+ [A-Z][a-z]+)",
    "present": r"(?i)(?:present|attending)[:\s]*([^\n]+)",
    "absent": r"(?i)absent[:\s]*([^\n]+)"
}
```

#### C. Decision/Motion Detection

```python
DECISION_PATTERNS = [
    r"(?i)motion (?:to|that) (.+?) (?:passed|carried|approved|failed|tabled)",
    r"(?i)(?:it was|the committee) (?:decided|agreed|resolved) (?:to |that )(.+)",
    r"(?i)(?:unanimous|majority) (?:decision|vote|agreement)[:.\s]*(.+)",
    r"(?i)vote[:\s]*(\d+)[- ](\d+)[- ](\d+)"  # For: Against: Abstain
]
```

#### D. Action Item Detection

```python
ACTION_PATTERNS = [
    r"(?i)action[:\s]+(.+?)(?:due|by|before)?[:\s]*(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)?",
    r"(?i)([A-Z][a-z]+ [A-Z][a-z]+) (?:will|to|should) (.+?)(?:\.|$)",
    r"(?i)(?:assigned to|responsible)[:\s]+([A-Z][a-z]+ [A-Z][a-z]+)",
    r"(?i)(?:deadline|due date|target)[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
]
```

#### E. Enhanced Entity Extraction with Relations

Instead of flat entity lists, extract triples:

```python
def extract_meeting_relations(doc):
    """Extract (subject, predicate, object) triples from meeting text."""
    relations = []

    for sent in doc.sents:
        # Find the main verb
        root = [t for t in sent if t.dep_ == "ROOT"][0] if any(t.dep_ == "ROOT" for t in sent) else None
        if not root:
            continue

        # Find subject (who)
        subjects = [t for t in sent if t.dep_ in ("nsubj", "nsubjpass")]

        # Find objects (what)
        objects = [t for t in sent if t.dep_ in ("dobj", "pobj", "attr")]

        for subj in subjects:
            for obj in objects:
                relations.append({
                    "subject": subj.text,
                    "predicate": root.lemma_,
                    "object": obj.text,
                    "sentence": sent.text
                })

    return relations
```

#### F. Coreference Resolution

Add coreference to link pronouns to named entities:

```python
# Using spacy-experimental or neuralcoref
import spacy
nlp = spacy.load("en_core_web_trf")  # Transformer model
nlp.add_pipe("coreferee")  # Or use neuralcoref

# Resolve: "Tom presented the report. He recommended approval."
# → "Tom presented the report. Tom recommended approval."
```

### Model Recommendations

| Task | Current Model | Recommended Model |
|------|---------------|-------------------|
| NER | en_core_web_sm | en_core_web_trf (transformer) |
| Dependency Parsing | en_core_web_sm | en_core_web_lg or trf |
| Coreference | None | coreferee or neuralcoref |
| Relation Extraction | None | Custom spaCy component |
| Pattern Matching | None | spaCy Matcher + EntityRuler |

---

## 4. Proposed Data Structure for Neo4j

### Node Types

```cypher
// Core nodes
(:Meeting {id, date, duration, quorum, location})
(:MeetingMinutes {id, name, approvalStatus, recordedBy, checksum, embedding})
(:Committee {id, name, type, established, dissolved})
(:Person {id, name, email, department})

// Enhanced nodes
(:Decision {id, text, type, outcome, voteTally})
(:ActionItem {id, description, dueDate, status, completedDate})
(:AgendaItem {id, title, sequence, discussionNotes})

// Role nodes (reified relationships)
(:CommitteeMembership {startDate, endDate, role})
(:MeetingParticipation {status, arrivalTime, departureTime})
```

### Relationship Types

```cypher
// Meeting relationships
(:Meeting)-[:PRODUCED]->(:MeetingMinutes)
(:Meeting)-[:HAS_AGENDA]->(:AgendaItem)
(:Meeting)-[:RESULTED_IN]->(:Decision)
(:Meeting)-[:CREATED]->(:ActionItem)
(:Meeting)-[:FOLLOWS]->(:Meeting)  // Previous meeting link
(:Meeting)-[:HELD_BY]->(:Committee)

// Participation (with role on relationship)
(:Person)-[:PARTICIPATED {role: "Chair", status: "Present"}]->(:Meeting)
(:Person)-[:RECORDED]->(:MeetingMinutes)
(:Person)-[:APPROVED]->(:MeetingMinutes)

// Action Items
(:ActionItem)-[:ASSIGNED_TO]->(:Person)
(:ActionItem)-[:REPORTED_IN]->(:Meeting)  // Follow-up mentions
(:ActionItem)-[:BLOCKED_BY]->(:ActionItem)

// Committee structure
(:Committee)-[:REPORTS_TO]->(:Committee)
(:Person)-[:MEMBER_OF {role, startDate, endDate}]->(:Committee)
(:Committee)-[:HAS_CHARTER]->(:Document)

// Document provenance
(:MeetingMinutes)-[:DERIVED_FROM]->(:MeetingMinutes)  // Versions
(:MeetingMinutes)-[:MENTIONS]->(:Person|Organization|Location)
```

### Example Query Patterns

```cypher
// Q: What decisions did the Executive Committee make about budget in 2023?
MATCH (c:Committee {name: "Executive Committee"})
      -[:HELD]-(m:Meeting)-[:RESULTED_IN]->(d:Decision)
WHERE m.date >= date("2023-01-01") AND m.date <= date("2023-12-31")
  AND toLower(d.text) CONTAINS "budget"
RETURN m.date, d.text, d.outcome

// Q: What action items are assigned to Tom Teper and still open?
MATCH (p:Person {name: "Tom Teper"})<-[:ASSIGNED_TO]-(a:ActionItem)
WHERE a.status = "Open"
RETURN a.description, a.dueDate, a.resultingFrom

// Q: Who attended the most meetings in 2022?
MATCH (p:Person)-[r:PARTICIPATED]->(m:Meeting)
WHERE m.date.year = 2022 AND r.status = "Present"
RETURN p.name, count(m) AS meetings
ORDER BY meetings DESC LIMIT 10

// Q: Trace decision provenance - who voted for budget increase?
MATCH (m:Meeting)-[:RESULTED_IN]->(d:Decision)-[:VOTED_FOR_BY]->(p:Person)
WHERE d.text CONTAINS "budget increase"
RETURN m.date, p.name, d.outcome
```

---

## 5. Improved JSON-LD Structure

### Before (Current)

```json
{
    "@context": {"@vocab": "http://schema.org/"},
    "@type": "CreativeWork",
    "name": "Executive Committee_Minutes_2022-05-23.docx",
    "creator": {"@type": "Organization", "name": "Library Staff"},
    "additionalType": "Minutes",
    "dateCreated": "2022-05-23",
    "entities": {
        "PERSON": ["Tom Teper", "John Wilkin"],
        "ORG": ["Executive Committee"],
        "DATE": ["May 23, 2022"]
    }
}
```

### After (Proposed)

```json
{
    "@context": {
        "@vocab": "http://schema.org/",
        "org": "http://www.w3.org/ns/org#",
        "prov": "http://www.w3.org/ns/prov#",
        "dcterms": "http://purl.org/dc/terms/",
        "lib": "http://library.example.org/ontology#"
    },
    "@type": ["lib:MeetingMinutes", "prov:Entity"],
    "@id": "urn:library:minutes:exec-2022-05-23",

    "name": "Executive Committee Minutes - May 23, 2022",
    "dateCreated": "2022-05-23",
    "fileFormat": "DOCX",

    "lib:approvalStatus": "Approved",
    "lib:approvedDate": "2022-06-06",

    "prov:wasGeneratedBy": {
        "@type": "lib:Meeting",
        "@id": "urn:library:meeting:exec-2022-05-23",
        "schema:startDate": "2022-05-23T14:00:00",
        "schema:endDate": "2022-05-23T15:30:00",
        "schema:location": "Main Library Conference Room",
        "lib:hasQuorum": true
    },

    "prov:wasAttributedTo": {
        "@type": "foaf:Person",
        "name": "Sarah Johnson",
        "lib:role": "Secretary"
    },

    "lib:attendance": [
        {
            "@type": "lib:MeetingParticipation",
            "lib:participant": {"@type": "foaf:Person", "name": "Tom Teper"},
            "lib:participationRole": "Chair",
            "lib:attendanceStatus": "Present"
        },
        {
            "@type": "lib:MeetingParticipation",
            "lib:participant": {"@type": "foaf:Person", "name": "John Wilkin"},
            "lib:participationRole": "Member",
            "lib:attendanceStatus": "Present"
        },
        {
            "@type": "lib:MeetingParticipation",
            "lib:participant": {"@type": "foaf:Person", "name": "Mary Smith"},
            "lib:participationRole": "Member",
            "lib:attendanceStatus": "Absent"
        }
    ],

    "lib:decisions": [
        {
            "@type": "lib:Decision",
            "@id": "urn:library:decision:exec-2022-05-23-001",
            "lib:decisionText": "Approve FY2023 budget allocation for digital resources",
            "lib:decisionType": "MotionPassed",
            "lib:voteOutcome": {"for": 5, "against": 0, "abstain": 1}
        }
    ],

    "lib:actionItems": [
        {
            "@type": "lib:ActionItem",
            "@id": "urn:library:action:exec-2022-05-23-001",
            "lib:actionDescription": "Prepare vendor comparison report",
            "lib:assignedTo": {"@type": "foaf:Person", "name": "John Wilkin"},
            "lib:dueDate": "2022-06-15",
            "lib:status": "Completed"
        }
    ],

    "lib:relatedCommittee": {
        "@type": ["org:FormalOrganization", "lib:Committee"],
        "@id": "urn:library:committee:executive",
        "name": "Executive Committee",
        "lib:committeeType": "Standing"
    },

    "lib:mentionedEntities": {
        "organizations": [
            {"@type": "org:Organization", "name": "HathiTrust"},
            {"@type": "org:Organization", "name": "CARLI"}
        ],
        "locations": [
            {"@type": "schema:Place", "name": "Main Library"}
        ],
        "topics": ["budget", "digital resources", "vendor selection"]
    },

    "checksum": {
        "@type": "schema:PropertyValue",
        "algorithm": "SHA-256",
        "value": "abc123..."
    },

    "prov:generatedAtTime": "2022-05-24T09:00:00Z"
}
```

---

## 6. Impact on RAG and Q&A Quality

### Current Limitations

The current GraphRAG implementation has limited context - only shows title, date, committee, mentioned people/orgs. Missing: roles, decisions, actions, relationships.

### With Enhanced Ontology

**Better Retrieval:**
- Query "What did Tom Teper decide about budget?" can now filter by:
  - Tom Teper's role (Chair decisions vs. discussed items)
  - Decision type (motions vs. consensus)
  - Vote outcomes

**Richer Context for LLM:**
```python
def format_enhanced_context(meeting_data):
    context = f"""
    Meeting: {meeting_data['committee']} on {meeting_data['date']}

    Attendees:
    - Chair: {meeting_data['chair']}
    - Present: {', '.join(meeting_data['present'])}
    - Absent: {', '.join(meeting_data['absent'])}

    Decisions Made:
    {format_decisions(meeting_data['decisions'])}

    Action Items Created:
    {format_actions(meeting_data['actions'])}

    Topics Discussed: {', '.join(meeting_data['topics'])}
    """
    return context
```

**New Query Types Enabled:**
- "What open action items does John have?"
- "List all decisions made about digital preservation"
- "Who voted against the budget proposal?"
- "What meetings did Tom chair in 2023?"
- "What was decided at the last Executive Committee meeting?"

---

## 7. Implementation Sequence

### Phase 1: Ontology Definition
1. Create formal OWL ontology file (`lib/ontology/committee-ontology.ttl`)
2. Define all classes, properties, and constraints
3. Create mapping from existing schema to new schema
4. Document with examples

### Phase 2: NLP Pipeline Enhancement
1. Add section detection module
2. Implement role-aware attendance extraction
3. Add decision/motion pattern detection
4. Add action item extraction
5. Implement basic relation extraction
6. Test on sample documents

### Phase 3: Data Migration
1. Create migration script to convert existing JSON-LD
2. Re-process documents with enhanced NLP
3. Validate migration with sample checks
4. Generate quality report

### Phase 4: Neo4j Schema Update
1. Design new Cypher schema
2. Create migration Cypher scripts
3. Update `neo4j_export.py` for new relationships
4. Test graph queries

### Phase 5: GraphRAG Enhancement
1. Update embeddings to include structured content
2. Modify retriever for new node types
3. Enhance context formatting for LLM
4. Add new query patterns

---

## 8. Critical Files for Implementation

1. **`data_pipeline/enhance_metadata.py`** - Core metadata creation logic; must be updated to generate enhanced JSON-LD structure with new ontology classes

2. **`data_pipeline/add_nlp_terms_to_metadata.py`** - NLP extraction pipeline; needs major enhancement for section detection, role extraction, decision/action detection

3. **`data_pipeline/neo4j_export.py`** - Graph export logic; must be refactored to create Meeting, Decision, ActionItem nodes and new relationship types

4. **`data_pipeline/graphrag_retriever.py`** - Retrieval queries; needs new Cypher patterns for enhanced graph traversal across meetings, decisions, and actions

5. **`data_pipeline/entity_validation.py`** - Entity filtering; needs expansion for role detection and relation validation

---

## References

- [W3C Organization Ontology (ORG)](https://www.w3.org/TR/vocab-org/)
- [PROV-O: The Provenance Ontology](https://www.w3.org/TR/prov-o/)
- [Records in Contexts Ontology (RiC-O)](https://www.ica.org/standards/RiC/ontology)
- [Dublin Core Metadata Initiative](https://www.dublincore.org/)
- [spaCy Dependency Parsing](https://spacy.io/api/dependencyparser)
- [Relation Extraction with spaCy](https://explosion.ai/blog/relation-extraction)

---

**Document prepared for:** IS547 Project Ontology Redesign
**Analysis date:** 2026-01-27
**Status:** Planning complete - ready for implementation