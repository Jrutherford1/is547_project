"""
NLP Quality Reporting Module

Tracks quality metrics during NLP extraction and generates reports
to identify problematic documents and rejected entities.
"""

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any


@dataclass
class DocumentQualityRecord:
    """Quality metrics for a single document."""
    file_path: str
    text_quality_score: float
    text_length: int
    extraction_success: bool
    issues: List[str] = field(default_factory=list)
    entities_extracted: Dict[str, int] = field(default_factory=dict)
    entities_rejected: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class QualityTracker:
    """
    Tracks quality metrics during batch NLP processing.

    Collects statistics on:
    - Document-level quality scores
    - Entity extraction and rejection counts
    - Reasons for entity rejection
    """

    def __init__(self):
        self.records: List[DocumentQualityRecord] = []
        self.rejected_entities: Dict[str, Counter] = {
            "PERSON": Counter(),
            "ORG": Counter(),
            "DATE": Counter(),
            "GPE": Counter()
        }
        self.rejection_reasons: Dict[str, Counter] = {
            "PERSON": Counter(),
            "ORG": Counter(),
            "DATE": Counter(),
            "GPE": Counter()
        }
        self.entity_totals: Dict[str, int] = {
            "PERSON": 0,
            "ORG": 0,
            "DATE": 0,
            "GPE": 0
        }
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    def start(self):
        """Mark the start of processing."""
        self.start_time = datetime.now()

    def stop(self):
        """Mark the end of processing."""
        self.end_time = datetime.now()

    def record_document(self, record: DocumentQualityRecord):
        """
        Record quality metrics for a processed document.

        Args:
            record: DocumentQualityRecord with metrics
        """
        self.records.append(record)

    def record_entity(self, entity_type: str, entity: str, accepted: bool, reason: str = ""):
        """
        Record an entity extraction result.

        Args:
            entity_type: Type of entity (PERSON, ORG, etc.)
            entity: The entity text
            accepted: Whether the entity was accepted
            reason: Reason for rejection (if rejected)
        """
        if entity_type not in self.entity_totals:
            return

        self.entity_totals[entity_type] += 1

        if not accepted:
            self.rejected_entities[entity_type][entity] += 1
            if reason:
                self.rejection_reasons[entity_type][reason] += 1

    def get_problematic_documents(self, threshold: float = 0.35) -> List[Dict]:
        """
        Get list of documents with quality score below threshold.

        Args:
            threshold: Quality score threshold

        Returns:
            List of document info dicts sorted by quality score
        """
        problematic = []
        for record in self.records:
            if record.text_quality_score < threshold:
                problematic.append({
                    'path': record.file_path,
                    'quality_score': record.text_quality_score,
                    'text_length': record.text_length,
                    'issues': record.issues
                })

        return sorted(problematic, key=lambda x: x['quality_score'])

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive quality report.

        Returns:
            Dict containing all quality metrics and statistics
        """
        self.stop()

        # Calculate summary statistics
        total_docs = len(self.records)
        successful = sum(1 for r in self.records if r.extraction_success)
        failed = total_docs - successful

        quality_scores = [r.text_quality_score for r in self.records if r.text_quality_score > 0]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

        low_quality_docs = self.get_problematic_documents()

        # Entity statistics
        entity_stats = {}
        for entity_type in ["PERSON", "ORG", "GPE", "DATE"]:
            total = self.entity_totals[entity_type]
            rejected = sum(self.rejected_entities[entity_type].values())
            entity_stats[entity_type] = {
                'total_extracted': total,
                'total_rejected': rejected,
                'total_kept': total - rejected,
                'rejection_rate': rejected / total if total > 0 else 0
            }

        # Top rejected entities
        top_rejected = {}
        for entity_type in ["PERSON", "ORG", "GPE", "DATE"]:
            top_rejected[entity_type] = [
                {'entity': entity, 'count': count}
                for entity, count in self.rejected_entities[entity_type].most_common(50)
            ]

        # Top rejection reasons
        rejection_reason_stats = {}
        for entity_type in ["PERSON", "ORG", "GPE", "DATE"]:
            rejection_reason_stats[entity_type] = dict(
                self.rejection_reasons[entity_type].most_common(20)
            )

        # All issues encountered
        all_issues = Counter()
        for record in self.records:
            for issue in record.issues:
                all_issues[issue] += 1

        # Processing time
        processing_time = None
        if self.start_time and self.end_time:
            processing_time = (self.end_time - self.start_time).total_seconds()

        report = {
            'timestamp': datetime.now().isoformat(),
            'processing_time_seconds': processing_time,
            'summary': {
                'total_documents': total_docs,
                'successful_extractions': successful,
                'failed_extractions': failed,
                'low_quality_documents': len(low_quality_docs),
                'avg_text_quality_score': round(avg_quality, 3)
            },
            'entity_stats': entity_stats,
            'top_rejected_entities': top_rejected,
            'rejection_reasons': rejection_reason_stats,
            'document_issues': dict(all_issues.most_common(20)),
            'problematic_documents': low_quality_docs[:100]  # Top 100 worst
        }

        return report

    def save_report(self, path: str) -> str:
        """
        Generate and save the quality report to a JSON file.

        Args:
            path: Output file path

        Returns:
            Path to saved file
        """
        report = self.generate_report()

        # Ensure directory exists
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return path

    def print_summary(self):
        """Print a summary of the quality metrics to console."""
        report = self.generate_report()

        print("\n" + "=" * 60)
        print("NLP EXTRACTION QUALITY REPORT")
        print("=" * 60)

        summary = report['summary']
        print(f"\nDocuments Processed: {summary['total_documents']}")
        print(f"  Successful: {summary['successful_extractions']}")
        print(f"  Failed: {summary['failed_extractions']}")
        print(f"  Low Quality: {summary['low_quality_documents']}")
        print(f"  Avg Quality Score: {summary['avg_text_quality_score']:.3f}")

        print("\nEntity Statistics:")
        for entity_type, stats in report['entity_stats'].items():
            print(f"  {entity_type}:")
            print(f"    Extracted: {stats['total_extracted']}")
            print(f"    Kept: {stats['total_kept']}")
            print(f"    Rejected: {stats['total_rejected']} ({stats['rejection_rate']:.1%})")

        print("\nTop Rejected PERSON Entities:")
        for item in report['top_rejected_entities']['PERSON'][:10]:
            entity_display = item['entity'][:40] + '...' if len(item['entity']) > 40 else item['entity']
            print(f"  '{entity_display}': {item['count']}")

        print("\nTop Rejection Reasons (PERSON):")
        for reason, count in list(report['rejection_reasons']['PERSON'].items())[:10]:
            print(f"  {reason}: {count}")

        if report['problematic_documents']:
            print(f"\nProblematic Documents (showing top 5 of {len(report['problematic_documents'])}):")
            for doc in report['problematic_documents'][:5]:
                print(f"  Score {doc['quality_score']:.2f}: {doc['path']}")
                if doc['issues']:
                    print(f"    Issues: {', '.join(doc['issues'][:3])}")

        print("\n" + "=" * 60)


def load_report(path: str) -> Dict[str, Any]:
    """
    Load a previously saved quality report.

    Args:
        path: Path to the JSON report file

    Returns:
        Report dictionary
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_reports(report1_path: str, report2_path: str) -> Dict[str, Any]:
    """
    Compare two quality reports to show improvement.

    Args:
        report1_path: Path to first (baseline) report
        report2_path: Path to second (improved) report

    Returns:
        Comparison dictionary
    """
    r1 = load_report(report1_path)
    r2 = load_report(report2_path)

    comparison = {
        'quality_improvement': {
            'avg_score_before': r1['summary']['avg_text_quality_score'],
            'avg_score_after': r2['summary']['avg_text_quality_score'],
            'low_quality_before': r1['summary']['low_quality_documents'],
            'low_quality_after': r2['summary']['low_quality_documents'],
        },
        'entity_improvements': {}
    }

    for entity_type in ["PERSON", "ORG", "GPE", "DATE"]:
        before = r1['entity_stats'].get(entity_type, {})
        after = r2['entity_stats'].get(entity_type, {})

        comparison['entity_improvements'][entity_type] = {
            'rejection_rate_before': before.get('rejection_rate', 0),
            'rejection_rate_after': after.get('rejection_rate', 0),
            'kept_before': before.get('total_kept', 0),
            'kept_after': after.get('total_kept', 0),
        }

    return comparison


if __name__ == "__main__":
    # Demo usage
    tracker = QualityTracker()
    tracker.start()

    # Simulate some records
    tracker.record_document(DocumentQualityRecord(
        file_path="test/doc1.pdf",
        text_quality_score=0.85,
        text_length=5000,
        extraction_success=True,
        issues=[],
        entities_extracted={"PERSON": 5, "ORG": 3},
        entities_rejected={"PERSON": 2, "ORG": 0}
    ))

    tracker.record_document(DocumentQualityRecord(
        file_path="test/doc2.pdf",
        text_quality_score=0.25,
        text_length=100,
        extraction_success=True,
        issues=["low_vowel_ratio", "low_dictionary_words"],
        entities_extracted={"PERSON": 10, "ORG": 5},
        entities_rejected={"PERSON": 8, "ORG": 3}
    ))

    # Simulate entity rejections
    tracker.record_entity("PERSON", "Tom Teper", True, "")
    tracker.record_entity("PERSON", "Library", False, "filter_term")
    tracker.record_entity("PERSON", "xdc garbage", False, "not_plausible_name")

    tracker.print_summary()
