import json
from datetime import datetime

def write_project_metadata(
    output_path="project_metadata.jsonld",
    dataset_name="Library Committee Documents Archive",
    description="Curated collection of meeting minutes, agendas, and related documents from library committees, enhanced with metadata and checksums.",
    creator="Library Staff",
    license_text="Open/Public per institutional policy",
    keywords=None
):
    if keywords is None:
        keywords = [
            "library governance",
            "committee records",
            "institutional memory",
            "metadata curation"
        ]

    dataset_metadata = {
        "@context": {
            "@vocab": "http://schema.org/"
        },
        "@type": "Dataset",
        "name": dataset_name,
        "description": description,
        "creator": {
            "@type": "Organization",
            "name": creator
        },
        "dateModified": datetime.now().strftime("%Y-%m-%d"),
        "license": license_text,
        "keywords": keywords,
        "hasPart": "Individual files described in subdirectory metadata",
        "isBasedOn": "Original documents migrated from legacy WordPress archive",
        "includedInDataCatalog": "Institutional repository or internal library system"
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset_metadata, f, indent=4)

    print(f"Project-level metadata written to: {output_path}")