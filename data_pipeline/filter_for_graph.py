"""
Filter documents for graph processing.

Creates a filtered dataset containing only Minutes documents,
organized by committee with a flattened structure.
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict


def scan_minutes_documents(source_dir: str | Path) -> dict[str, list[tuple[Path, Path]]]:
    """
    Scan Processed_Committees for Minutes documents.

    Args:
        source_dir: Path to Processed_Committees directory

    Returns:
        Dict mapping committee names to list of (doc_path, json_path) tuples
    """
    source_path = Path(source_dir)
    committees = defaultdict(list)

    # Iterate through committee directories
    for committee_dir in sorted(source_path.iterdir()):
        if not committee_dir.is_dir():
            continue

        committee_name = committee_dir.name
        minutes_dir = committee_dir / "Minutes"

        if not minutes_dir.exists() or not minutes_dir.is_dir():
            continue

        # Find all JSON files in Minutes folder
        for json_file in minutes_dir.glob("*.json"):
            # Verify it's actually a Minutes document
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                if metadata.get("additionalType") != "Minutes":
                    continue
            except (json.JSONDecodeError, IOError):
                continue

            # Find the matching document file
            base_name = json_file.stem  # filename without .json
            doc_path = None

            for file in minutes_dir.iterdir():
                if file.is_file() and file.stem == base_name and file.suffix != '.json':
                    doc_path = file
                    break

            if doc_path:
                committees[committee_name].append((doc_path, json_file))

    return dict(committees)


def copy_minutes_to_graph_folder(
    source_dir: str | Path,
    dest_dir: str | Path,
    verbose: bool = True
) -> dict:
    """
    Copy Minutes documents to a flattened graph folder structure.

    Args:
        source_dir: Path to Processed_Committees directory
        dest_dir: Destination directory for filtered documents
        verbose: Print progress information

    Returns:
        Stats dict with committees_copied, documents_copied, skipped_empty
    """
    dest_path = Path(dest_dir)

    # Clean destination if it exists
    if dest_path.exists():
        if verbose:
            print(f"Removing existing directory: {dest_path}")
        shutil.rmtree(dest_path)

    dest_path.mkdir(parents=True, exist_ok=True)

    # Scan for minutes documents
    if verbose:
        print(f"Scanning for Minutes documents in: {source_dir}")

    committees_data = scan_minutes_documents(source_dir)

    stats = {
        "committees_copied": 0,
        "documents_copied": 0,
        "skipped_empty": 0,
        "committees_with_minutes": []
    }

    # Count empty committees (those not in committees_data)
    source_path = Path(source_dir)
    all_committees = [d.name for d in source_path.iterdir() if d.is_dir()]
    stats["skipped_empty"] = len(all_committees) - len(committees_data)

    # Copy files for each committee
    for committee_name, doc_pairs in sorted(committees_data.items()):
        if not doc_pairs:
            stats["skipped_empty"] += 1
            continue

        # Create committee directory (flattened - no Minutes subfolder)
        committee_dest = dest_path / committee_name
        committee_dest.mkdir(parents=True, exist_ok=True)

        for doc_path, json_path in doc_pairs:
            # Copy document
            shutil.copy2(doc_path, committee_dest / doc_path.name)
            # Copy JSON metadata
            shutil.copy2(json_path, committee_dest / json_path.name)
            stats["documents_copied"] += 1

        stats["committees_copied"] += 1
        stats["committees_with_minutes"].append(committee_name)

        if verbose:
            print(f"  {committee_name}: {len(doc_pairs)} documents")

    return stats


def filter_for_graph(
    source_dir: str | Path = "data/Processed_Committees",
    dest_dir: str | Path = "data/committees_processed_for_graph"
) -> dict:
    """
    Main entry point for filtering documents for graph processing.

    Creates a filtered dataset containing only Minutes documents,
    organized by committee with a flattened structure.

    Args:
        source_dir: Path to Processed_Committees directory
        dest_dir: Destination directory for filtered documents

    Returns:
        Stats dict with processing results
    """
    print("=" * 60)
    print("Filtering Documents for Graph Processing")
    print("=" * 60)
    print(f"Source: {source_dir}")
    print(f"Destination: {dest_dir}")
    print()

    stats = copy_minutes_to_graph_folder(source_dir, dest_dir, verbose=True)

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Committees with minutes: {stats['committees_copied']}")
    print(f"Committees skipped (no minutes): {stats['skipped_empty']}")
    print(f"Documents copied: {stats['documents_copied']}")
    print(f"Total files: {stats['documents_copied'] * 2} (documents + JSON metadata)")
    print(f"Output directory: {dest_dir}")
    print()

    return stats


if __name__ == "__main__":
    filter_for_graph()
