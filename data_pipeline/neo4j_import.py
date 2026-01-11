"""
Neo4j Direct Import Module for Library Committee Documents

Connects directly to Neo4j and imports the committee document metadata.
Uses credentials from .env file.

Usage:
    from data_pipeline.neo4j_import import import_to_neo4j
    import_to_neo4j()
"""

import os
import json
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()


class Neo4jImporter:
    """Imports committee document metadata directly into Neo4j."""

    # Generic terms to filter from PERSON entities
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

    def __init__(self, uri: str = None, user: str = None, password: str = None):
        """Initialize connection to Neo4j."""
        self.uri = uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.user = user or os.getenv('NEO4J_USER', 'neo4j')
        self.password = password or os.getenv('NEO4J_PASSWORD')

        if not self.password:
            raise ValueError("NEO4J_PASSWORD not found in environment or .env file")

        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

        # Verify connection
        self.driver.verify_connectivity()
        print(f"Connected to Neo4j at {self.uri}")

    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()

    def _clean_person_name(self, name: str) -> Optional[str]:
        """Clean and validate person names."""
        if not name or not isinstance(name, str):
            return None
        name = ' '.join(name.split())
        if len(name) < 2 or len(name) > 100:
            return None
        name_lower = name.lower()
        for term in self.PERSON_FILTER_TERMS:
            if term in name_lower:
                return None
        if name == name.lower():
            return None
        return name

    def clear_database(self):
        """Clear all nodes and relationships from the database."""
        print("Clearing existing data...")
        with self.driver.session() as session:
            # Delete all relationships first, then nodes
            session.run("MATCH ()-[r]->() DELETE r")
            session.run("MATCH (n) DELETE n")
            # Drop existing constraints (ignore errors if they don't exist)
            for constraint in ['doc_id', 'committee_name', 'person_name', 'org_name', 'location_name']:
                try:
                    session.run(f"DROP CONSTRAINT {constraint} IF EXISTS")
                except:
                    pass
        print("Database cleared.")

    def create_constraints(self):
        """Create uniqueness constraints and indexes."""
        print("Creating constraints and indexes...")
        with self.driver.session() as session:
            constraints = [
                "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT committee_name IF NOT EXISTS FOR (c:Committee) REQUIRE c.name IS UNIQUE",
                "CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
                "CREATE CONSTRAINT org_name IF NOT EXISTS FOR (o:Organization) REQUIRE o.name IS UNIQUE",
                "CREATE CONSTRAINT location_name IF NOT EXISTS FOR (l:Location) REQUIRE l.name IS UNIQUE",
                "CREATE INDEX doc_date IF NOT EXISTS FOR (d:Document) ON (d.dateCreated)",
                "CREATE INDEX doc_type IF NOT EXISTS FOR (d:Document) ON (d.additionalType)",
            ]
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    print(f"  Warning: {e}")
        print("Constraints created.")

    def import_data(self, base_dir: str = "data/Processed_Committees",
                    limit: Optional[int] = None,
                    min_person_mentions: int = 2,
                    batch_size: int = 100):
        """
        Import all committee document data into Neo4j.

        Args:
            base_dir: Base directory containing processed committee files
            limit: Optional limit on number of files to process
            min_person_mentions: Minimum mentions to include a person
            batch_size: Number of operations per transaction
        """
        print(f"Scanning documents in {base_dir}...")

        # First pass: collect all data
        documents = {}
        committees = set()
        persons = defaultdict(int)
        organizations = defaultdict(int)
        locations = defaultdict(int)
        doc_person = []
        doc_org = []
        doc_location = []
        person_coappears = defaultdict(int)

        processed = 0
        for root, _, files in os.walk(base_dir):
            json_files = [f for f in files if f.endswith(".json")]
            for json_file in json_files:
                if limit and processed >= limit:
                    break

                json_path = os.path.join(root, json_file)
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)

                    # Generate IDs and extract data
                    rel_path = os.path.relpath(json_path, base_dir)
                    doc_id = os.path.splitext(rel_path)[0].replace(os.sep, "__")
                    committee = rel_path.split(os.sep)[0]

                    # Derive file path
                    file_format = metadata.get('fileFormat', 'unknown').lower()
                    json_base = os.path.splitext(json_path)[0]
                    doc_file_path = os.path.abspath(f"{json_base}.{file_format}")

                    documents[doc_id] = {
                        'id': doc_id,
                        'name': metadata.get('name', json_file),
                        'dateCreated': metadata.get('dateCreated', 'unknown'),
                        'fileFormat': metadata.get('fileFormat', 'unknown'),
                        'additionalType': metadata.get('additionalType', 'unknown'),
                        'description': metadata.get('description', ''),
                        'checksum': metadata.get('checksum', {}).get('value', ''),
                        'originalFileName': metadata.get('originalFileName', ''),
                        'filePath': doc_file_path,
                        'committee': committee
                    }
                    committees.add(committee)

                    # Process entities
                    entities = metadata.get('entities', {})
                    persons_in_doc = []

                    for person in entities.get('PERSON', []):
                        clean_name = self._clean_person_name(person)
                        if clean_name:
                            persons[clean_name] += 1
                            doc_person.append((doc_id, clean_name))
                            persons_in_doc.append(clean_name)

                    # Co-appearances
                    for i, p1 in enumerate(persons_in_doc):
                        for p2 in persons_in_doc[i+1:]:
                            pair = tuple(sorted([p1, p2]))
                            person_coappears[pair] += 1

                    for org in entities.get('ORG', []):
                        if org and len(org) > 1:
                            organizations[org] += 1
                            doc_org.append((doc_id, org))

                    for loc in entities.get('GPE', []):
                        if loc and len(loc) > 1:
                            locations[loc] += 1
                            doc_location.append((doc_id, loc))

                    processed += 1
                    if processed % 500 == 0:
                        print(f"  Scanned {processed} documents...")

                except Exception as e:
                    print(f"  Error reading {json_path}: {e}")

            if limit and processed >= limit:
                break

        print(f"\nScanned {len(documents)} documents")
        print(f"  Committees: {len(committees)}")
        print(f"  Persons: {len(persons)}")
        print(f"  Organizations: {len(organizations)}")
        print(f"  Locations: {len(locations)}")

        # Filter persons by min mentions
        important_persons = {p for p, c in persons.items() if c >= min_person_mentions}
        print(f"  Persons (≥{min_person_mentions} mentions): {len(important_persons)}")

        # Import to Neo4j
        with self.driver.session() as session:
            # Create committees
            print("\nImporting committees...")
            for committee in committees:
                session.run("MERGE (c:Committee {name: $name})", name=committee)

            # Create persons
            print("Importing persons...")
            for person, count in persons.items():
                if count >= min_person_mentions:
                    session.run(
                        "MERGE (p:Person {name: $name}) SET p.mentionCount = $count",
                        name=person, count=count
                    )

            # Create organizations
            print("Importing organizations...")
            for org, count in organizations.items():
                session.run(
                    "MERGE (o:Organization {name: $name}) SET o.mentionCount = $count",
                    name=org, count=count
                )

            # Create locations
            print("Importing locations...")
            for loc, count in locations.items():
                session.run(
                    "MERGE (l:Location {name: $name}) SET l.mentionCount = $count",
                    name=loc, count=count
                )

            # Create documents with BELONGS_TO relationships
            print("Importing documents...")
            count = 0
            for doc_id, props in documents.items():
                session.run("""
                    MERGE (d:Document {id: $id})
                    SET d.name = $name,
                        d.dateCreated = $dateCreated,
                        d.fileFormat = $fileFormat,
                        d.additionalType = $additionalType,
                        d.description = $description,
                        d.checksum = $checksum,
                        d.originalFileName = $originalFileName,
                        d.filePath = $filePath
                    WITH d
                    MATCH (c:Committee {name: $committee})
                    MERGE (d)-[:BELONGS_TO]->(c)
                """, **props)
                count += 1
                if count % 500 == 0:
                    print(f"  Imported {count} documents...")
            print(f"  Imported {count} documents total")

            # Create MENTIONS relationships for persons
            print("Creating person mentions...")
            count = 0
            for doc_id, person in doc_person:
                if person in important_persons:
                    session.run("""
                        MATCH (d:Document {id: $doc_id}), (p:Person {name: $person})
                        MERGE (d)-[:MENTIONS]->(p)
                    """, doc_id=doc_id, person=person)
                    count += 1
            print(f"  Created {count} person mentions")

            # Create MENTIONS relationships for organizations
            print("Creating organization mentions...")
            count = 0
            for doc_id, org in doc_org:
                session.run("""
                    MATCH (d:Document {id: $doc_id}), (o:Organization {name: $org})
                    MERGE (d)-[:MENTIONS]->(o)
                """, doc_id=doc_id, org=org)
                count += 1
            print(f"  Created {count} organization mentions")

            # Create MENTIONS relationships for locations
            print("Creating location mentions...")
            count = 0
            for doc_id, loc in doc_location:
                session.run("""
                    MATCH (d:Document {id: $doc_id}), (l:Location {name: $loc})
                    MERGE (d)-[:MENTIONS]->(l)
                """, doc_id=doc_id, loc=loc)
                count += 1
            print(f"  Created {count} location mentions")

            # Create CO_APPEARS_WITH relationships
            print("Creating co-appearance relationships...")
            count = 0
            for (p1, p2), cocount in person_coappears.items():
                if p1 in important_persons and p2 in important_persons:
                    session.run("""
                        MATCH (p1:Person {name: $p1}), (p2:Person {name: $p2})
                        MERGE (p1)-[r:CO_APPEARS_WITH]->(p2)
                        SET r.count = $count
                    """, p1=p1, p2=p2, count=cocount)
                    count += 1
            print(f"  Created {count} co-appearance relationships")

        print("\nImport complete!")

    def get_stats(self) -> dict:
        """Get statistics about the current graph."""
        stats = {}
        with self.driver.session() as session:
            stats['documents'] = session.run("MATCH (d:Document) RETURN count(d) AS count").single()['count']
            stats['committees'] = session.run("MATCH (c:Committee) RETURN count(c) AS count").single()['count']
            stats['persons'] = session.run("MATCH (p:Person) RETURN count(p) AS count").single()['count']
            stats['organizations'] = session.run("MATCH (o:Organization) RETURN count(o) AS count").single()['count']
            stats['locations'] = session.run("MATCH (l:Location) RETURN count(l) AS count").single()['count']
            stats['relationships'] = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()['count']
        return stats


def import_to_neo4j(base_dir: str = "data/Processed_Committees",
                    limit: Optional[int] = None,
                    min_person_mentions: int = 2,
                    clear_first: bool = True) -> dict:
    """
    Main function to import committee documents into Neo4j.

    Args:
        base_dir: Base directory containing processed committee files
        limit: Optional limit on number of files to process
        min_person_mentions: Minimum mentions to include a person
        clear_first: Whether to clear the database before importing

    Returns:
        Dict with graph statistics
    """
    importer = Neo4jImporter()

    try:
        if clear_first:
            importer.clear_database()

        importer.create_constraints()
        importer.import_data(
            base_dir=base_dir,
            limit=limit,
            min_person_mentions=min_person_mentions
        )

        stats = importer.get_stats()
        print("\n=== Graph Statistics ===")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        return stats

    finally:
        importer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Import committee documents to Neo4j")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of documents to process")
    parser.add_argument("--min-mentions", type=int, default=2,
                        help="Minimum mentions to include a person (default: 2)")
    parser.add_argument("--no-clear", action="store_true",
                        help="Don't clear database before importing")

    args = parser.parse_args()

    import_to_neo4j(
        limit=args.limit,
        min_person_mentions=args.min_mentions,
        clear_first=not args.no_clear
    )
