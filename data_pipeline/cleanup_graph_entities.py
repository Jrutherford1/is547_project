"""
Cleanup and reprocess NLP entities for the graph dataset.

This module processes the filtered minutes-only dataset in
data/committees_processed_for_graph/ with stricter entity validation
to improve quality for knowledge graph and GraphRAG applications.

Created: 2026-01-26
Purpose: Provenance-tracked entity cleanup separate from main pipeline
"""

import os
import json
from collections import defaultdict, Counter
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import warnings
import logging

import spacy

# Load spaCy model once at module level
nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger"])

# Import text extraction
from data_pipeline.nlp_term_extraction_preview import extract_text

# Import quality validation modules
from data_pipeline.text_quality import calculate_text_quality_score
from data_pipeline.entity_validation import validate_entity, clean_entity_text
from data_pipeline.nlp_quality_report import QualityTracker, DocumentQualityRecord


# Additional filter terms for stricter cleaning
ADDITIONAL_PERSON_FILTERS = {
    # First names only (too ambiguous)
    "chris", "victor", "jen", "joe", "bill", "john", "tom", "jim",
    "mike", "bob", "dave", "steve", "dan", "mark", "paul", "scott",
    "mary", "lisa", "sarah", "jennifer", "jessica", "amanda", "melissa",
    "lauren", "zoe", "lynne", "elisabeth", "mara", "sara", "clara",
    "francisco", "heather", "kelli", "hannah", "sue", "robert",

    # Non-persons that slip through
    "gateway", "primo", "carli", "ideals", "sfx", "faq", "bipoc",
    "microsoft teams", "teams", "box", "aeon", "archives",

    # Acronyms and abbreviations
    "ec", "deia", "tf", "ap", "capt", "ugl", "cms", "ac", "ala",
    "appit", "hr", "uiuc",
}

# ORG entities to filter out
ORG_FILTER_TERMS = {
    # Short acronyms that lack context
    "ec", "ap", "ac", "tf", "hr", "it", "ga", "pr", "pm", "qa",
    "ugl", "cdc", "cms", "cic", "aul", "arl", "ala", "prc",
    "appit", "capt", "deia", "uiuc", "carli",

    # Generic terms
    "committee", "attendees", "university", "divisions", "the executive committee",
    "the committee", "the university", "members", "staff", "group",
    "task force", "the task force", "working group", "team",
    "invited attendees", "budget", "division", "senate",

    # Software/systems (not orgs)
    "primo", "ideals", "sfx", "libnews", "cms", "teams", "box",
    "microsoft teams", "google", "zoom", "webex", "libguides", "archon",
    "powerpoint", "libguide", "camtasia", "facebook", "twitter",

    # Misclassified
    "ideas", "n't", "libraries",

    # Known misclassified person names in ORG
    "cherié weible", "cherie weible", "marek sroka", "jameatris rimkus",
    "wordpress",
}

# GPE (location) entities to filter out
GPE_FILTER_TERMS = {
    # Misclassified person names (common first names appearing as GPE)
    "merinda", "cindy", "lucretia", "esra", "atoma", "paula",
    "kirstin", "joann", "lori", "tina", "beth", "greg",
    "becky", "qiang", "jameatris", "shuyong", "ariana", "laila",

    # Not locations
    "prc", "n't", "libraries", "ideas", "primo", "archon", "box",
    "inclusion", "the mortenson center", "mortenson center",
    "ns corridor", "library 428", "new business", "systems", "mpal",
    "main", "d. ward",

    # Too generic
    "oak street", "oak st.", "oak st",
}

# Known person name patterns (for detecting misclassified persons in ORG/GPE)
COMMON_FIRST_NAMES = {
    "john", "mary", "james", "patricia", "robert", "jennifer", "michael",
    "linda", "william", "elizabeth", "david", "barbara", "richard", "susan",
    "joseph", "jessica", "thomas", "sarah", "charles", "karen", "chris",
    "christopher", "daniel", "nancy", "matthew", "lisa", "anthony", "betty",
    "mark", "margaret", "donald", "sandra", "steven", "ashley", "paul",
    "kimberly", "andrew", "emily", "joshua", "donna", "kenneth", "michelle",
    "kevin", "dorothy", "brian", "carol", "george", "amanda", "timothy",
    "melissa", "ronald", "deborah", "edward", "stephanie", "jason", "rebecca",
    "jeffrey", "sharon", "ryan", "laura", "jacob", "cynthia", "gary", "kathleen",
    "nicholas", "amy", "eric", "angela", "jonathan", "shirley", "stephen",
    "anna", "larry", "brenda", "justin", "pamela", "scott", "emma", "brandon",
    "nicole", "benjamin", "helen", "samuel", "samantha", "gregory", "katherine",
    "frank", "christine", "alexander", "debra", "patrick", "rachel", "raymond",
    "carolyn", "jack", "janet", "dennis", "catherine", "jerry", "maria",
    "tyler", "heather", "aaron", "diane", "jose", "ruth", "adam", "julie",
    "nathan", "olivia", "henry", "joyce", "douglas", "virginia", "zachary",
    "victoria", "peter", "kelly", "kyle", "lauren", "noah", "christina",
    # Additional names from our data
    "tom", "bill", "sue", "jim", "joe", "jen", "mike", "bob", "dave",
    "steve", "dan", "sara", "clara", "mara", "lynne", "kelli", "hannah",
    "zoe", "cindy", "merinda", "lucretia", "esra", "atoma", "paula",
    "kirstin", "joann", "lori", "tina", "beth", "greg", "victor",
}


def is_valid_person_name(name: str) -> Tuple[bool, str]:
    """
    Additional validation for PERSON entities with stricter rules.

    Args:
        name: The entity text to validate

    Returns:
        Tuple of (is_valid, rejection_reason)
    """
    name_lower = name.lower().strip()

    # Check against additional filters
    if name_lower in ADDITIONAL_PERSON_FILTERS:
        return False, "additional_filter"

    # Reject single words (likely first names only)
    words = name.split()
    if len(words) == 1:
        # Allow if it looks like a full name with period (e.g., "J. Smith")
        if '.' not in name:
            return False, "single_word_name"

    # Reject if all uppercase (likely acronym)
    if name.isupper() and len(name) <= 10:
        return False, "all_caps_short"

    # Reject if contains "the " at start (likely not a person)
    if name_lower.startswith("the "):
        return False, "starts_with_the"

    return True, ""


def is_valid_org(name: str) -> Tuple[bool, str]:
    """
    Validate ORG entities with stricter rules.

    Filters out:
    - Short acronyms (2-4 chars, all caps)
    - Generic organizational terms
    - Things that look like person names
    - Software/system names
    - Overly long phrases

    Args:
        name: The entity text to validate

    Returns:
        Tuple of (is_valid, rejection_reason)
    """
    name_lower = name.lower().strip()

    # Check against ORG filter terms
    if name_lower in ORG_FILTER_TERMS:
        return False, "org_filter_term"

    # Reject short all-caps acronyms (2-5 characters)
    if name.isupper() and len(name) <= 5:
        return False, "short_acronym"

    # Reject if it looks like a person name (FirstName LastName pattern)
    words = name.split()
    if len(words) == 2:
        first, last = words
        if first.lower() in COMMON_FIRST_NAMES:
            # Check if last name is capitalized like a name
            if len(last) > 1 and last[0].isupper() and last[1:].islower():
                return False, "looks_like_person_name"

    # Reject single common first names
    if len(words) == 1 and name_lower in COMMON_FIRST_NAMES:
        return False, "single_first_name"

    # Reject if starts with "the " (often generic)
    if name_lower.startswith("the "):
        return False, "starts_with_the"

    # Reject overly long ORG names (likely sentence fragments)
    if len(words) > 7:
        return False, "too_many_words"

    # Reject if contains "time and location" or similar meeting artifacts
    if "time and location" in name_lower or "meeting" in name_lower:
        return False, "meeting_artifact"

    # Reject if has unbalanced parentheses (partial text fragments)
    if name.count('(') != name.count(')'):
        return False, "unbalanced_parens"

    # Reject if ends with a comma (partial text)
    if name.endswith(','):
        return False, "ends_with_comma"

    return True, ""


def is_valid_gpe(name: str) -> Tuple[bool, str]:
    """
    Validate GPE (geopolitical entity/location) entities with stricter rules.

    Filters out:
    - Misclassified person names
    - Non-location terms
    - Generic or ambiguous terms

    Args:
        name: The entity text to validate

    Returns:
        Tuple of (is_valid, rejection_reason)
    """
    name_lower = name.lower().strip()

    # Check against GPE filter terms
    if name_lower in GPE_FILTER_TERMS:
        return False, "gpe_filter_term"

    # Reject if it looks like a person name
    words = name.split()

    # Single word that's a common first name
    if len(words) == 1 and name_lower in COMMON_FIRST_NAMES:
        return False, "looks_like_first_name"

    # Two words that look like "FirstName LastName" pattern
    if len(words) == 2:
        first, last = words
        # Check if first word is a known first name
        if first.lower() in COMMON_FIRST_NAMES:
            if len(last) > 1 and last[0].isupper() and last[1:].islower():
                return False, "looks_like_person_name"
        # Also check for capitalized two-word patterns that look like names
        # (catches uncommon names like "Shuyong Jiang", "Jameatris Rimkus")
        if (len(first) > 1 and first[0].isupper() and first[1:].islower() and
            len(last) > 1 and last[0].isupper() and last[1:].islower()):
            # Looks like "Firstname Lastname" - probably a person, not a place
            # Real places are usually "City Name" or single words
            # Check if neither word is a common place word
            place_words = {"city", "town", "county", "state", "north", "south",
                          "east", "west", "new", "old", "port", "fort", "mount",
                          "lake", "river", "bay", "island", "spring", "falls"}
            if first.lower() not in place_words and last.lower() not in place_words:
                return False, "two_word_name_pattern"

    # Reject very short entries (likely errors)
    if len(name) <= 2:
        return False, "too_short"

    # Reject if contains apostrophe-t (contraction artifact) or is just "n't"
    # U+2019 is RIGHT SINGLE QUOTATION MARK (curly apostrophe)
    curly_apos = "\u2019"
    if (f"n{curly_apos}t" in name or "n't" in name or
        f"{curly_apos}t" in name or "'t" in name or
        name == f"n{curly_apos}t" or name == "n't"):
        return False, "contraction_artifact"

    # Also reject if the name is very short and contains apostrophe-like characters
    if len(name) <= 3 and ("'" in name or curly_apos in name):
        return False, "short_with_apostrophe"

    # Reject non-location words that slip through
    non_gpe_terms = {"provost", "requesting", "camtasia", "powerpoint", "libguide"}
    if name_lower in non_gpe_terms:
        return False, "not_a_location"

    # Reject if starts with "the " (usually not a valid GPE)
    if name_lower.startswith("the "):
        return False, "starts_with_the"

    # Reject if contains numbers (e.g., "Library 428")
    if any(c.isdigit() for c in name):
        return False, "contains_numbers"

    return True, ""


def extract_entities_strict(
    file_paths: List[str],
    quality_tracker: Optional[QualityTracker] = None,
    text_quality_threshold: float = 0.35
) -> List[Tuple[Dict[str, List[str]], Dict[str, List[Tuple[str, str]]], float]]:
    """
    Extract entities with strict validation for graph dataset.

    Args:
        file_paths: List of document file paths
        quality_tracker: Optional tracker for quality metrics
        text_quality_threshold: Minimum quality score for text

    Returns:
        List of (entities_dict, rejected_dict, quality_score) tuples
    """
    texts = []
    valid_paths = []
    quality_scores = []

    for file_path in file_paths:
        try:
            text = extract_text(file_path)
            if text and len(text.strip()) > 0:
                quality = calculate_text_quality_score(text)
                quality_scores.append(quality['overall_score'])

                if quality['is_valid'] and quality['overall_score'] >= text_quality_threshold:
                    texts.append(text)
                else:
                    texts.append("")
                    if quality_tracker:
                        quality_tracker.record_document(DocumentQualityRecord(
                            file_path=file_path,
                            text_quality_score=quality['overall_score'],
                            text_length=len(text),
                            extraction_success=False,
                            issues=quality['issues'],
                            entities_extracted={},
                            entities_rejected={}
                        ))
                valid_paths.append(file_path)
            else:
                texts.append("")
                valid_paths.append(file_path)
                quality_scores.append(0.0)
        except Exception as e:
            texts.append("")
            valid_paths.append(file_path)
            quality_scores.append(0.0)

    # Batch process with spaCy
    results = []
    for doc, file_path, quality_score in zip(
        nlp.pipe(texts, batch_size=50), valid_paths, quality_scores
    ):
        entities = {"PERSON": [], "ORG": [], "GPE": [], "DATE": []}
        rejected = {"PERSON": [], "ORG": [], "GPE": [], "DATE": []}

        for ent in doc.ents:
            if ent.label_ not in entities:
                continue

            cleaned = clean_entity_text(ent.text)
            if not cleaned:
                continue

            # Standard validation
            is_valid, reason = validate_entity(cleaned, ent.label_)

            # Additional strict validation by entity type
            if is_valid and ent.label_ == "PERSON":
                is_valid, reason = is_valid_person_name(cleaned)
            elif is_valid and ent.label_ == "ORG":
                is_valid, reason = is_valid_org(cleaned)
            elif is_valid and ent.label_ == "GPE":
                is_valid, reason = is_valid_gpe(cleaned)

            if is_valid:
                if cleaned not in entities[ent.label_]:
                    entities[ent.label_].append(cleaned)
                    if quality_tracker:
                        quality_tracker.record_entity(ent.label_, cleaned, True, "")
            else:
                rejected[ent.label_].append((cleaned, reason))
                if quality_tracker:
                    quality_tracker.record_entity(ent.label_, cleaned, False, reason)

        # Sort entity lists
        for key in entities:
            entities[key] = sorted(entities[key])

        results.append((entities, rejected, quality_score))

    return results


def cleanup_graph_entities(
    base_dir: str = "data/committees_processed_for_graph",
    batch_size: int = 100,
    generate_report: bool = True,
    report_path: str = "data/graph_nlp_quality_report.json",
    text_quality_threshold: float = 0.35
) -> Dict[str, Any]:
    """
    Clean up and reprocess all entities in the graph dataset.

    This clears existing entities and reprocesses with stricter validation
    suitable for knowledge graph and GraphRAG applications.

    Args:
        base_dir: Directory containing the filtered graph dataset
        batch_size: Files to process per batch
        generate_report: Whether to generate quality report
        report_path: Path for quality report JSON
        text_quality_threshold: Minimum text quality score

    Returns:
        Dict with processing statistics
    """
    # Suppress PDF warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="pdfplumber")
    logging.getLogger("pdfplumber").setLevel(logging.ERROR)

    print("=" * 60)
    print("CLEANING UP GRAPH DATASET ENTITIES")
    print("=" * 60)
    print(f"Source: {base_dir}")
    print(f"Report: {report_path}")
    print()

    # Initialize tracker
    quality_tracker = QualityTracker() if generate_report else None
    if quality_tracker:
        quality_tracker.start()

    # Stats
    stats = {
        "files_examined": 0,
        "files_processed": 0,
        "files_updated": 0,
        "files_skipped": 0,
        "entities": {"PERSON": Counter(), "ORG": Counter(), "GPE": Counter(), "DATE": Counter()},
        "rejected_counts": {"PERSON": 0, "ORG": 0, "GPE": 0, "DATE": 0}
    }

    supported_extensions = [".txt", ".docx", ".pptx", ".pdf", ".doc", ".ppt"]

    # Step 1: Clear existing entities
    print("Step 1: Clearing existing entities...")
    cleared = 0
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith('.json'):
                json_path = os.path.join(root, f)
                try:
                    with open(json_path, 'r', encoding='utf-8') as file:
                        metadata = json.load(file)

                    if 'entities' in metadata:
                        metadata['entities'] = {}
                    if 'keywords' in metadata:
                        metadata['keywords'] = []

                    with open(json_path, 'w', encoding='utf-8') as file:
                        json.dump(metadata, file, indent=4)
                    cleared += 1
                except Exception as e:
                    print(f"  Warning: Could not clear {f}: {e}")

    print(f"  Cleared {cleared} JSON files")

    # Step 2: Collect files to process
    print("\nStep 2: Collecting files to process...")
    files_to_process = []

    for root, _, files in os.walk(base_dir):
        json_files = [f for f in files if f.endswith('.json')]

        for json_file in json_files:
            json_path = os.path.join(root, json_file)
            base_name = Path(json_file).stem
            stats["files_examined"] += 1

            # Find corresponding document
            for ext in supported_extensions:
                doc_path = os.path.join(root, base_name + ext)
                if os.path.exists(doc_path):
                    committee = os.path.basename(root)
                    files_to_process.append((doc_path, json_path, committee))
                    break

    print(f"  Found {len(files_to_process)} document-JSON pairs")

    # Step 3: Process in batches
    print("\nStep 3: Processing with strict entity validation...")

    for i in range(0, len(files_to_process), batch_size):
        batch = files_to_process[i:i + batch_size]
        doc_paths = [item[0] for item in batch]

        try:
            batch_results = extract_entities_strict(
                doc_paths,
                quality_tracker=quality_tracker,
                text_quality_threshold=text_quality_threshold
            )

            for (doc_path, json_path, committee), (entities, rejected, quality_score) in zip(batch, batch_results):
                try:
                    # Update entity counters
                    for entity_type, entity_list in entities.items():
                        for entity in entity_list:
                            stats["entities"][entity_type][entity] += 1

                    # Update rejected counters
                    for entity_type, rejected_list in rejected.items():
                        stats["rejected_counts"][entity_type] += len(rejected_list)

                    # Load and update JSON
                    with open(json_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)

                    metadata["entities"] = {}
                    for entity_type, entity_list in entities.items():
                        if entity_list:
                            metadata["entities"][entity_type] = entity_list

                    # Update keywords
                    keywords = []
                    if entities["PERSON"]:
                        keywords.extend(entities["PERSON"][:3])
                    if entities["ORG"]:
                        keywords.extend(entities["ORG"][:3])
                    metadata["keywords"] = keywords

                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=4)

                    stats["files_updated"] += 1
                    stats["files_processed"] += 1

                except Exception as e:
                    stats["files_skipped"] += 1

            # Progress update
            processed = i + len(batch)
            if processed % 200 == 0 or processed == len(files_to_process):
                print(f"  Processed {processed}/{len(files_to_process)} files...")

        except Exception as e:
            print(f"  Batch error: {e}")
            stats["files_skipped"] += len(batch)

    # Summary
    print("\n" + "=" * 60)
    print("CLEANUP COMPLETE")
    print("=" * 60)
    print(f"Files examined: {stats['files_examined']}")
    print(f"Files processed: {stats['files_processed']}")
    print(f"Files updated: {stats['files_updated']}")
    print(f"Files skipped: {stats['files_skipped']}")

    print(f"\nEntities extracted (unique):")
    for entity_type, counter in stats["entities"].items():
        print(f"  {entity_type}: {len(counter)}")

    print(f"\nEntities rejected:")
    for entity_type, count in stats["rejected_counts"].items():
        print(f"  {entity_type}: {count}")

    # Generate report
    if quality_tracker and generate_report:
        quality_tracker.save_report(report_path)
        quality_tracker.print_summary()
        print(f"\nQuality report saved to: {report_path}")

    return stats


def show_top_entities(
    base_dir: str = "data/committees_processed_for_graph",
    top_n: int = 30
) -> None:
    """
    Display top entities from the graph dataset after cleanup.

    Args:
        base_dir: Directory containing the graph dataset
        top_n: Number of top entities to show
    """
    entities = {"PERSON": Counter(), "ORG": Counter(), "GPE": Counter(), "DATE": Counter()}

    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith('.json'):
                json_path = os.path.join(root, f)
                try:
                    with open(json_path, 'r', encoding='utf-8') as file:
                        metadata = json.load(file)

                    for entity_type, entity_list in metadata.get("entities", {}).items():
                        if entity_type in entities:
                            for entity in entity_list:
                                entities[entity_type][entity] += 1
                except Exception:
                    pass

    print("=" * 60)
    print(f"TOP {top_n} ENTITIES IN GRAPH DATASET")
    print("=" * 60)

    for entity_type, counter in entities.items():
        print(f"\n=== {entity_type} ({len(counter)} unique) ===")
        for name, count in counter.most_common(top_n):
            print(f"  {count:4d}  {name}")


if __name__ == "__main__":
    # Run cleanup
    result = cleanup_graph_entities()

    # Show results
    print("\n")
    show_top_entities()
