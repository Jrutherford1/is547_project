"""
Neo4j Export Module for Library Committee Documents

Generates Cypher statements and/or CSV files for importing the committee
document metadata into Neo4j.

Graph Model:
- (:Document) - Committee documents with metadata
- (:Committee) - Library committees
- (:Person) - People mentioned in documents
- (:Organization) - Organizations mentioned
- (:Location) - Geographic locations (GPE entities)

Relationships:
- (:Document)-[:BELONGS_TO]->(:Committee)
- (:Document)-[:MENTIONS]->(:Person|Organization|Location)
- (:Person)-[:CO_APPEARS_WITH]->(:Person) - People in same document
"""

import os
import json
import csv
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from datetime import datetime


class Neo4jExporter:
    """Exports committee document metadata to Neo4j format."""

    # Generic terms to filter from PERSON entities (inherited from existing pipeline)
    PERSON_FILTER_TERMS = {
        'library', 'staff', 'committee', 'group', 'team', 'division',
        'department', 'unit', 'office', 'council', 'board', 'meeting',
        'agenda', 'minutes', 'note', 'notes', 'action', 'item', 'items',
        'update', 'report', 'review', 'discussion', 'motion', 'vote',
        'present', 'absent', 'attendee', 'attendees', 'member', 'members',
        'chair', 'co-chair', 'secretary', 'treasurer', 'dean', 'director',
        'box', 'training', 'maps', 'numbers', 'archon', 'solberg', 'sousa',
        'krannert', 'hort'
    }

    def __init__(self, base_dir: str = "data/Processed_Committees"):
        self.base_dir = base_dir

        # Track unique entities with their properties
        self.documents: Dict[str, dict] = {}  # doc_id -> properties
        self.committees: Set[str] = set()
        self.persons: Dict[str, int] = defaultdict(int)  # name -> mention count
        self.organizations: Dict[str, int] = defaultdict(int)
        self.locations: Dict[str, int] = defaultdict(int)

        # Track relationships
        self.doc_committee: List[Tuple[str, str]] = []  # (doc_id, committee)
        self.doc_person: List[Tuple[str, str]] = []     # (doc_id, person)
        self.doc_org: List[Tuple[str, str]] = []        # (doc_id, org)
        self.doc_location: List[Tuple[str, str]] = []   # (doc_id, location)
        self.person_coappears: Dict[Tuple[str, str], int] = defaultdict(int)  # (p1, p2) -> count

    def _clean_person_name(self, name: str) -> Optional[str]:
        """Clean and validate person names."""
        if not name or not isinstance(name, str):
            return None

        # Remove newlines and extra whitespace
        name = ' '.join(name.split())

        # Skip if too short or too long
        if len(name) < 2 or len(name) > 100:
            return None

        # Skip if contains filter terms
        name_lower = name.lower()
        for term in self.PERSON_FILTER_TERMS:
            if term in name_lower:
                return None

        # Skip if all lowercase (likely not a proper name)
        if name == name.lower():
            return None

        return name

    def _sanitize_for_cypher(self, text: str) -> str:
        """Escape special characters for Cypher strings."""
        if not text:
            return ""
        return text.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')

    def _generate_doc_id(self, json_path: str) -> str:
        """Generate a unique document ID from the file path."""
        # Use relative path from base_dir as ID
        rel_path = os.path.relpath(json_path, self.base_dir)
        # Remove .json extension
        doc_id = os.path.splitext(rel_path)[0]
        # Replace path separators with underscores for cleaner IDs
        doc_id = doc_id.replace(os.sep, "__")
        return doc_id

    def _extract_committee_from_path(self, json_path: str) -> str:
        """Extract committee name from file path."""
        rel_path = os.path.relpath(json_path, self.base_dir)
        parts = rel_path.split(os.sep)
        return parts[0] if parts else "Unknown"

    def scan_documents(self, limit: Optional[int] = None, verbose: bool = True):
        """
        Scan all JSON metadata files and collect entities/relationships.

        Args:
            limit: Optional limit on number of files to process
            verbose: Print progress messages
        """
        processed = 0

        if verbose:
            print(f"Scanning documents in {self.base_dir}...")

        for root, _, files in os.walk(self.base_dir):
            json_files = [f for f in files if f.endswith(".json")]

            for json_file in json_files:
                if limit and processed >= limit:
                    break

                json_path = os.path.join(root, json_file)

                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)

                    self._process_document(json_path, metadata)
                    processed += 1

                    if verbose and processed % 500 == 0:
                        print(f"  Processed {processed} documents...")

                except Exception as e:
                    if verbose:
                        print(f"  Error processing {json_path}: {e}")

            if limit and processed >= limit:
                break

        if verbose:
            print(f"\nScan complete:")
            print(f"  Documents: {len(self.documents)}")
            print(f"  Committees: {len(self.committees)}")
            print(f"  Persons: {len(self.persons)}")
            print(f"  Organizations: {len(self.organizations)}")
            print(f"  Locations: {len(self.locations)}")
            print(f"  Co-appearances: {len(self.person_coappears)}")

    def _process_document(self, json_path: str, metadata: dict):
        """Process a single document's metadata."""
        doc_id = self._generate_doc_id(json_path)
        committee = self._extract_committee_from_path(json_path)

        # Derive actual document file path from JSON metadata path
        file_format = metadata.get('fileFormat', 'unknown').lower()
        json_base = os.path.splitext(json_path)[0]  # Remove .json
        doc_file_path = f"{json_base}.{file_format}"
        # Convert to absolute path
        doc_file_path = os.path.abspath(doc_file_path)

        # Extract document properties
        self.documents[doc_id] = {
            'name': metadata.get('name', os.path.basename(json_path)),
            'dateCreated': metadata.get('dateCreated', 'unknown'),
            'fileFormat': metadata.get('fileFormat', 'unknown'),
            'additionalType': metadata.get('additionalType', 'unknown'),
            'description': metadata.get('description', ''),
            'checksum': metadata.get('checksum', {}).get('value', ''),
            'originalFileName': metadata.get('originalFileName', ''),
            'filePath': doc_file_path,
            'committee': committee
        }

        # Track committee
        self.committees.add(committee)
        self.doc_committee.append((doc_id, committee))

        # Process entities
        entities = metadata.get('entities', {})

        # Process PERSON entities
        persons_in_doc = []
        for person in entities.get('PERSON', []):
            clean_name = self._clean_person_name(person)
            if clean_name:
                self.persons[clean_name] += 1
                self.doc_person.append((doc_id, clean_name))
                persons_in_doc.append(clean_name)

        # Build co-appearance relationships
        for i, p1 in enumerate(persons_in_doc):
            for p2 in persons_in_doc[i+1:]:
                # Always store in sorted order for consistency
                pair = tuple(sorted([p1, p2]))
                self.person_coappears[pair] += 1

        # Process ORG entities
        for org in entities.get('ORG', []):
            if org and len(org) > 1:
                self.organizations[org] += 1
                self.doc_org.append((doc_id, org))

        # Process GPE (location) entities
        for loc in entities.get('GPE', []):
            if loc and len(loc) > 1:
                self.locations[loc] += 1
                self.doc_location.append((doc_id, loc))

    def generate_cypher(self, output_file: str = "data/neo4j_export/neo4j_import.cypher",
                        min_person_mentions: int = 2,
                        min_coappear_count: int = 2) -> str:
        """
        Generate Cypher CREATE statements for Neo4j import.

        Args:
            output_file: Path to output .cypher file
            min_person_mentions: Minimum mentions to include a person
            min_coappear_count: Minimum co-appearances for relationship

        Returns:
            Path to generated file
        """
        lines = []

        # Header
        lines.append("// Neo4j Import Script for Library Committee Documents")
        lines.append(f"// Generated: {datetime.now().isoformat()}")
        lines.append(f"// Documents: {len(self.documents)}")
        lines.append(f"// Committees: {len(self.committees)}")
        lines.append("")

        # Create constraints/indexes first (for performance)
        lines.append("// Create constraints and indexes")
        lines.append("CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;")
        lines.append("CREATE CONSTRAINT committee_name IF NOT EXISTS FOR (c:Committee) REQUIRE c.name IS UNIQUE;")
        lines.append("CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE;")
        lines.append("CREATE CONSTRAINT org_name IF NOT EXISTS FOR (o:Organization) REQUIRE o.name IS UNIQUE;")
        lines.append("CREATE CONSTRAINT location_name IF NOT EXISTS FOR (l:Location) REQUIRE l.name IS UNIQUE;")
        lines.append("CREATE INDEX doc_date IF NOT EXISTS FOR (d:Document) ON (d.dateCreated);")
        lines.append("CREATE INDEX doc_type IF NOT EXISTS FOR (d:Document) ON (d.additionalType);")
        lines.append("")

        # Create Committee nodes
        lines.append("// Create Committee nodes")
        for committee in sorted(self.committees):
            safe_name = self._sanitize_for_cypher(committee)
            lines.append(f"MERGE (c:Committee {{name: '{safe_name}'}});")
        lines.append("")

        # Create Person nodes (filtered by min mentions)
        lines.append(f"// Create Person nodes (min {min_person_mentions} mentions)")
        for person, count in sorted(self.persons.items()):
            if count >= min_person_mentions:
                safe_name = self._sanitize_for_cypher(person)
                lines.append(f"MERGE (p:Person {{name: '{safe_name}'}}) SET p.mentionCount = {count};")
        lines.append("")

        # Create Organization nodes
        lines.append("// Create Organization nodes")
        for org, count in sorted(self.organizations.items()):
            safe_name = self._sanitize_for_cypher(org)
            lines.append(f"MERGE (o:Organization {{name: '{safe_name}'}}) SET o.mentionCount = {count};")
        lines.append("")

        # Create Location nodes
        lines.append("// Create Location nodes")
        for loc, count in sorted(self.locations.items()):
            safe_name = self._sanitize_for_cypher(loc)
            lines.append(f"MERGE (l:Location {{name: '{safe_name}'}}) SET l.mentionCount = {count};")
        lines.append("")

        # Create Document nodes and BELONGS_TO relationships
        lines.append("// Create Document nodes with BELONGS_TO relationships")
        for doc_id, props in self.documents.items():
            safe_id = self._sanitize_for_cypher(doc_id)
            safe_name = self._sanitize_for_cypher(props['name'])
            safe_desc = self._sanitize_for_cypher(props['description'])
            safe_orig = self._sanitize_for_cypher(props['originalFileName'])
            safe_path = self._sanitize_for_cypher(props['filePath'])
            safe_committee = self._sanitize_for_cypher(props['committee'])

            lines.append(
                f"MERGE (d:Document {{id: '{safe_id}'}}) "
                f"SET d.name = '{safe_name}', "
                f"d.dateCreated = '{props['dateCreated']}', "
                f"d.fileFormat = '{props['fileFormat']}', "
                f"d.additionalType = '{props['additionalType']}', "
                f"d.description = '{safe_desc}', "
                f"d.checksum = '{props['checksum']}', "
                f"d.originalFileName = '{safe_orig}', "
                f"d.filePath = '{safe_path}' "
                f"WITH d "
                f"MATCH (c:Committee {{name: '{safe_committee}'}}) "
                f"MERGE (d)-[:BELONGS_TO]->(c);"
            )
        lines.append("")

        # Create MENTIONS relationships for Persons
        lines.append(f"// Create MENTIONS relationships (Person, min {min_person_mentions} mentions)")
        important_persons = {p for p, c in self.persons.items() if c >= min_person_mentions}
        for doc_id, person in self.doc_person:
            if person in important_persons:
                safe_doc = self._sanitize_for_cypher(doc_id)
                safe_person = self._sanitize_for_cypher(person)
                lines.append(
                    f"MATCH (d:Document {{id: '{safe_doc}'}}), (p:Person {{name: '{safe_person}'}}) "
                    f"MERGE (d)-[:MENTIONS]->(p);"
                )
        lines.append("")

        # Create MENTIONS relationships for Organizations
        lines.append("// Create MENTIONS relationships (Organization)")
        for doc_id, org in self.doc_org:
            safe_doc = self._sanitize_for_cypher(doc_id)
            safe_org = self._sanitize_for_cypher(org)
            lines.append(
                f"MATCH (d:Document {{id: '{safe_doc}'}}), (o:Organization {{name: '{safe_org}'}}) "
                f"MERGE (d)-[:MENTIONS]->(o);"
            )
        lines.append("")

        # Create MENTIONS relationships for Locations
        lines.append("// Create MENTIONS relationships (Location)")
        for doc_id, loc in self.doc_location:
            safe_doc = self._sanitize_for_cypher(doc_id)
            safe_loc = self._sanitize_for_cypher(loc)
            lines.append(
                f"MATCH (d:Document {{id: '{safe_doc}'}}), (l:Location {{name: '{safe_loc}'}}) "
                f"MERGE (d)-[:MENTIONS]->(l);"
            )
        lines.append("")

        # Create CO_APPEARS_WITH relationships
        lines.append(f"// Create CO_APPEARS_WITH relationships (min {min_coappear_count} co-appearances)")
        for (p1, p2), count in sorted(self.person_coappears.items(), key=lambda x: -x[1]):
            if count >= min_coappear_count and p1 in important_persons and p2 in important_persons:
                safe_p1 = self._sanitize_for_cypher(p1)
                safe_p2 = self._sanitize_for_cypher(p2)
                lines.append(
                    f"MATCH (p1:Person {{name: '{safe_p1}'}}), (p2:Person {{name: '{safe_p2}'}}) "
                    f"MERGE (p1)-[:CO_APPEARS_WITH {{count: {count}}}]->(p2);"
                )
        lines.append("")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"Cypher script written to: {output_file}")
        return output_file

    def generate_csv(self, output_dir: str = "data/neo4j_export/csv",
                     min_person_mentions: int = 2) -> str:
        """
        Generate CSV files for neo4j-admin import (faster for large datasets).

        Args:
            output_dir: Directory to write CSV files
            min_person_mentions: Minimum mentions to include a person

        Returns:
            Path to output directory
        """
        os.makedirs(output_dir, exist_ok=True)

        # Committees CSV
        with open(os.path.join(output_dir, 'committees.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['name:ID(Committee)'])
            for committee in sorted(self.committees):
                writer.writerow([committee])

        # Persons CSV
        with open(os.path.join(output_dir, 'persons.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['name:ID(Person)', 'mentionCount:int'])
            for person, count in sorted(self.persons.items()):
                if count >= min_person_mentions:
                    writer.writerow([person, count])

        # Organizations CSV
        with open(os.path.join(output_dir, 'organizations.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['name:ID(Organization)', 'mentionCount:int'])
            for org, count in sorted(self.organizations.items()):
                writer.writerow([org, count])

        # Locations CSV
        with open(os.path.join(output_dir, 'locations.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['name:ID(Location)', 'mentionCount:int'])
            for loc, count in sorted(self.locations.items()):
                writer.writerow([loc, count])

        # Documents CSV
        with open(os.path.join(output_dir, 'documents.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id:ID(Document)', 'name', 'dateCreated', 'fileFormat',
                           'additionalType', 'description', 'checksum', 'originalFileName', 'filePath'])
            for doc_id, props in self.documents.items():
                writer.writerow([
                    doc_id,
                    props['name'],
                    props['dateCreated'],
                    props['fileFormat'],
                    props['additionalType'],
                    props['description'],
                    props['checksum'],
                    props['originalFileName'],
                    props['filePath']
                ])

        # Relationships: Document -> Committee
        with open(os.path.join(output_dir, 'doc_belongs_to_committee.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([':START_ID(Document)', ':END_ID(Committee)', ':TYPE'])
            for doc_id, committee in self.doc_committee:
                writer.writerow([doc_id, committee, 'BELONGS_TO'])

        # Relationships: Document -> Person
        important_persons = {p for p, c in self.persons.items() if c >= min_person_mentions}
        with open(os.path.join(output_dir, 'doc_mentions_person.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([':START_ID(Document)', ':END_ID(Person)', ':TYPE'])
            for doc_id, person in self.doc_person:
                if person in important_persons:
                    writer.writerow([doc_id, person, 'MENTIONS'])

        # Relationships: Document -> Organization
        with open(os.path.join(output_dir, 'doc_mentions_org.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([':START_ID(Document)', ':END_ID(Organization)', ':TYPE'])
            for doc_id, org in self.doc_org:
                writer.writerow([doc_id, org, 'MENTIONS'])

        # Relationships: Document -> Location
        with open(os.path.join(output_dir, 'doc_mentions_location.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([':START_ID(Document)', ':END_ID(Location)', ':TYPE'])
            for doc_id, loc in self.doc_location:
                writer.writerow([doc_id, loc, 'MENTIONS'])

        # Relationships: Person -> Person (co-appears)
        with open(os.path.join(output_dir, 'person_coappears.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([':START_ID(Person)', ':END_ID(Person)', ':TYPE', 'count:int'])
            for (p1, p2), count in self.person_coappears.items():
                if p1 in important_persons and p2 in important_persons:
                    writer.writerow([p1, p2, 'CO_APPEARS_WITH', count])

        print(f"CSV files written to: {output_dir}/")
        print("Files created:")
        for f in os.listdir(output_dir):
            print(f"  - {f}")

        return output_dir


def export_to_neo4j(base_dir: str = "data/Processed_Committees",
                    output_format: str = "both",
                    limit: Optional[int] = None,
                    min_person_mentions: int = 2,
                    min_coappear_count: int = 2) -> dict:
    """
    Main function to export committee document data to Neo4j format.

    Args:
        base_dir: Base directory containing processed committee files
        output_format: "cypher", "csv", or "both"
        limit: Optional limit on number of files to process
        min_person_mentions: Minimum mentions to include a person
        min_coappear_count: Minimum co-appearances for CO_APPEARS_WITH

    Returns:
        Dict with paths to generated files
    """
    exporter = Neo4jExporter(base_dir)
    exporter.scan_documents(limit=limit)

    result = {}

    if output_format in ("cypher", "both"):
        result['cypher'] = exporter.generate_cypher(
            min_person_mentions=min_person_mentions,
            min_coappear_count=min_coappear_count
        )

    if output_format in ("csv", "both"):
        result['csv_dir'] = exporter.generate_csv(
            min_person_mentions=min_person_mentions
        )

    return result


# Sample queries to include in output
SAMPLE_QUERIES = """
// ============================================
// SAMPLE NEO4J CYPHER QUERIES
// ============================================

// 1. Find all documents for a specific committee
MATCH (d:Document)-[:BELONGS_TO]->(c:Committee {name: 'Executive Committee'})
RETURN d.name, d.dateCreated, d.additionalType
ORDER BY d.dateCreated DESC;

// 2. Find people who appear in the most documents
MATCH (p:Person)<-[:MENTIONS]-(d:Document)
RETURN p.name, count(d) AS docCount
ORDER BY docCount DESC
LIMIT 20;

// 3. Find which committees a specific person appears in
MATCH (p:Person {name: 'John Wilkin'})<-[:MENTIONS]-(d:Document)-[:BELONGS_TO]->(c:Committee)
RETURN DISTINCT c.name, count(d) AS docCount
ORDER BY docCount DESC;

// 4. Find people who frequently appear together
MATCH (p1:Person)-[r:CO_APPEARS_WITH]->(p2:Person)
RETURN p1.name, p2.name, r.count AS coAppearances
ORDER BY coAppearances DESC
LIMIT 20;

// 5. Find the path connecting two committees via shared people
MATCH path = (c1:Committee {name: 'Executive Committee'})<-[:BELONGS_TO]-(d1:Document)
             -[:MENTIONS]->(p:Person)<-[:MENTIONS]-(d2:Document)-[:BELONGS_TO]->(c2:Committee)
WHERE c1 <> c2
RETURN DISTINCT c2.name, p.name, count(*) AS connections
ORDER BY connections DESC
LIMIT 10;

// 6. Timeline: documents by year
MATCH (d:Document)
WHERE d.dateCreated <> 'unknown'
RETURN substring(d.dateCreated, 0, 4) AS year, count(*) AS docCount
ORDER BY year;

// 7. Find organizations mentioned alongside a specific person
MATCH (p:Person {name: 'Bill Maher'})<-[:MENTIONS]-(d:Document)-[:MENTIONS]->(o:Organization)
RETURN DISTINCT o.name, count(*) AS coMentions
ORDER BY coMentions DESC;

// 8. Community detection - find clusters of people who work together
CALL gds.louvain.stream({
    nodeProjection: 'Person',
    relationshipProjection: 'CO_APPEARS_WITH'
})
YIELD nodeId, communityId
RETURN gds.util.asNode(nodeId).name AS person, communityId
ORDER BY communityId, person;

// 9. PageRank - find most influential people in the network
CALL gds.pageRank.stream({
    nodeProjection: 'Person',
    relationshipProjection: 'CO_APPEARS_WITH'
})
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS person, score
ORDER BY score DESC
LIMIT 10;

// 10. Document type distribution per committee
MATCH (d:Document)-[:BELONGS_TO]->(c:Committee)
RETURN c.name, d.additionalType, count(*) AS count
ORDER BY c.name, count DESC;
"""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export committee documents to Neo4j format")
    parser.add_argument("--format", choices=["cypher", "csv", "both"], default="both",
                        help="Output format (default: both)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of documents to process")
    parser.add_argument("--min-mentions", type=int, default=2,
                        help="Minimum mentions to include a person (default: 2)")
    parser.add_argument("--min-coappear", type=int, default=2,
                        help="Minimum co-appearances for relationship (default: 2)")

    args = parser.parse_args()

    result = export_to_neo4j(
        output_format=args.format,
        limit=args.limit,
        min_person_mentions=args.min_mentions,
        min_coappear_count=args.min_coappear
    )

    print("\n" + "="*50)
    print("Export complete!")
    print("="*50)

    if 'cypher' in result:
        print(f"\nCypher script: {result['cypher']}")
        print("  Import with: cat neo4j_import.cypher | cypher-shell -u neo4j -p <password>")

    if 'csv_dir' in result:
        print(f"\nCSV files: {result['csv_dir']}/")
        print("  Import with neo4j-admin import (see Neo4j docs)")

    print("\n" + SAMPLE_QUERIES)