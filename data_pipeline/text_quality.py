"""
Text Quality Validation Module

Provides heuristics to detect garbage/corrupted text before NLP processing.
This is particularly useful for catching corrupted PDF extractions.
"""

import re
from typing import Dict, List, Tuple
from collections import Counter


# Common English words for dictionary check (minimal set for speed)
# These are the most frequent words that appear in most documents
COMMON_WORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
    'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
    'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
    'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
    'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other',
    'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
    'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
    'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us',
    'is', 'are', 'was', 'were', 'been', 'being', 'has', 'had', 'does', 'did',
    'meeting', 'committee', 'library', 'staff', 'discussion', 'update', 'report',
    'agenda', 'minutes', 'motion', 'vote', 'present', 'absent', 'action', 'item',
    'review', 'member', 'members', 'chair', 'next', 'last', 'please', 'thank',
    'project', 'plan', 'budget', 'space', 'building', 'service', 'services',
    'collection', 'collections', 'digital', 'access', 'public', 'information'
}

VOWELS = set('aeiouAEIOU')
CONSONANTS = set('bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ')


def calculate_vowel_ratio(text: str) -> float:
    """
    Calculate the ratio of vowels to total letters.

    English typically has ~38-42% vowels.
    Garbage text often has extreme ratios (<20% or >55%).

    Args:
        text: Input text to analyze

    Returns:
        Vowel ratio (0.0 to 1.0)
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    vowel_count = sum(1 for c in letters if c in VOWELS)
    return vowel_count / len(letters)


def calculate_punctuation_ratio(text: str) -> float:
    """
    Calculate the ratio of punctuation to total characters.

    Normal text: 5-15% punctuation.
    Garbage often has >25% unusual characters.

    Args:
        text: Input text to analyze

    Returns:
        Punctuation ratio (0.0 to 1.0)
    """
    if not text:
        return 0.0
    # Count non-alphanumeric, non-space characters
    punct_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return punct_count / len(text)


def calculate_word_length_stats(text: str) -> Dict[str, float]:
    """
    Calculate word length statistics.

    Returns mean, median, and max word length.
    Garbage text often has unusual distributions.

    Args:
        text: Input text to analyze

    Returns:
        Dict with 'mean', 'median', 'max' word lengths
    """
    words = re.findall(r'\b\w+\b', text)
    if not words:
        return {'mean': 0.0, 'median': 0.0, 'max': 0}

    lengths = [len(w) for w in words]
    sorted_lengths = sorted(lengths)
    n = len(sorted_lengths)

    return {
        'mean': sum(lengths) / len(lengths),
        'median': sorted_lengths[n // 2] if n % 2 else (sorted_lengths[n//2 - 1] + sorted_lengths[n//2]) / 2,
        'max': max(lengths)
    }


def calculate_dictionary_ratio(text: str) -> float:
    """
    Calculate ratio of words that match common English words.

    Normal text: >60% recognizable words.
    Garbage: <30% recognizable words.

    Args:
        text: Input text to analyze

    Returns:
        Dictionary word ratio (0.0 to 1.0)
    """
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if not words:
        return 0.0

    matches = sum(1 for w in words if w in COMMON_WORDS)
    return matches / len(words)


def detect_repeated_patterns(text: str) -> float:
    """
    Detect repeated character sequences that indicate corruption.

    OCR garbage often has repeated gibberish patterns.

    Args:
        text: Input text to analyze

    Returns:
        Score 0.0-1.0 where higher means more repetition detected
    """
    if len(text) < 10:
        return 0.0

    # Look for repeated 3-character sequences
    trigrams = [text[i:i+3] for i in range(len(text) - 2)]
    if not trigrams:
        return 0.0

    trigram_counts = Counter(trigrams)
    # Count trigrams that appear more than twice
    repeated = sum(count - 1 for count in trigram_counts.values() if count > 2)

    return min(1.0, repeated / len(trigrams))


def calculate_digit_ratio(text: str) -> float:
    """
    Calculate ratio of digits to total characters.

    Args:
        text: Input text to analyze

    Returns:
        Digit ratio (0.0 to 1.0)
    """
    if not text:
        return 0.0
    digit_count = sum(1 for c in text if c.isdigit())
    return digit_count / len(text)


def calculate_uppercase_ratio(text: str) -> float:
    """
    Calculate ratio of uppercase letters to total letters.

    Args:
        text: Input text to analyze

    Returns:
        Uppercase ratio (0.0 to 1.0)
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    upper_count = sum(1 for c in letters if c.isupper())
    return upper_count / len(letters)


def calculate_text_quality_score(text: str) -> Dict:
    """
    Comprehensive text quality analysis.

    Analyzes text and returns quality metrics to determine if
    the text is likely garbage/corrupted.

    Args:
        text: Input text to analyze

    Returns:
        Dict containing:
        - overall_score: 0.0-1.0 (1.0 = high quality)
        - is_valid: bool (meets minimum threshold)
        - metrics: detailed breakdown
        - issues: list of detected problems
    """
    if not text or len(text.strip()) < 10:
        return {
            'overall_score': 0.0,
            'is_valid': False,
            'metrics': {},
            'issues': ['text_too_short']
        }

    issues = []
    scores = []

    # Calculate all metrics
    vowel_ratio = calculate_vowel_ratio(text)
    punct_ratio = calculate_punctuation_ratio(text)
    word_stats = calculate_word_length_stats(text)
    dict_ratio = calculate_dictionary_ratio(text)
    repeat_score = detect_repeated_patterns(text)
    digit_ratio = calculate_digit_ratio(text)
    upper_ratio = calculate_uppercase_ratio(text)

    metrics = {
        'vowel_ratio': vowel_ratio,
        'punctuation_ratio': punct_ratio,
        'word_length_mean': word_stats['mean'],
        'word_length_max': word_stats['max'],
        'dictionary_ratio': dict_ratio,
        'repetition_score': repeat_score,
        'digit_ratio': digit_ratio,
        'uppercase_ratio': upper_ratio,
        'text_length': len(text)
    }

    # Score each metric (0 = bad, 1 = good)

    # Vowel ratio: expect 0.30-0.50
    if vowel_ratio < 0.20:
        issues.append('low_vowel_ratio')
        scores.append(vowel_ratio / 0.20)  # Partial credit
    elif vowel_ratio > 0.55:
        issues.append('high_vowel_ratio')
        scores.append(max(0, 1 - (vowel_ratio - 0.55) / 0.15))
    else:
        scores.append(1.0)

    # Punctuation ratio: expect < 0.20
    if punct_ratio > 0.30:
        issues.append('high_punctuation')
        scores.append(max(0, 1 - (punct_ratio - 0.20) / 0.30))
    elif punct_ratio > 0.20:
        scores.append(0.7)
    else:
        scores.append(1.0)

    # Word length: expect mean 3-10, penalize heavily outside this range
    if word_stats['mean'] < 2:
        issues.append('very_short_words')
        scores.append(0.1)
    elif word_stats['mean'] > 12:
        issues.append('very_long_words')
        # Heavily penalize very long average word length (sign of OCR garbage)
        scores.append(max(0, 0.5 - (word_stats['mean'] - 12) / 10))
    elif word_stats['mean'] > 10:
        scores.append(0.7)
    else:
        scores.append(1.0)

    # Dictionary ratio: expect > 0.20 - critical metric for garbage detection
    if dict_ratio < 0.05:
        issues.append('very_low_dictionary_words')
        scores.append(0.0)  # Zero score for almost no recognizable words
    elif dict_ratio < 0.15:
        issues.append('low_dictionary_words')
        scores.append(0.2)  # Heavy penalty
    elif dict_ratio < 0.25:
        scores.append(0.5)
    else:
        scores.append(1.0)

    # Repetition: expect < 0.15
    if repeat_score > 0.25:
        issues.append('high_repetition')
        scores.append(max(0, 1 - repeat_score))
    else:
        scores.append(1.0)

    # Digit ratio: expect < 0.20 for text documents
    if digit_ratio > 0.40:
        issues.append('high_digit_ratio')
        scores.append(0.5)
    else:
        scores.append(1.0)

    # Calculate overall score (weighted average)
    # Dictionary and word length are most important for detecting garbage
    weights = [1.0, 0.8, 2.0, 3.0, 1.0, 0.5]  # dict=3.0, word_length=2.0
    overall_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)

    # Automatic fail conditions (regardless of overall score)
    if dict_ratio < 0.05 and len(text) > 20:
        overall_score = min(overall_score, 0.15)
    if word_stats['mean'] > 15:
        overall_score = min(overall_score, 0.25)

    # Determine validity threshold
    is_valid = overall_score >= 0.35 and 'very_low_dictionary_words' not in issues

    return {
        'overall_score': round(overall_score, 3),
        'is_valid': is_valid,
        'metrics': metrics,
        'issues': issues
    }


def is_garbage_text(text: str, threshold: float = 0.35) -> bool:
    """
    Quick check if text appears to be garbage/corrupted.

    Args:
        text: Input text to analyze
        threshold: Quality score threshold (default 0.35)

    Returns:
        True if text appears to be garbage, False otherwise
    """
    result = calculate_text_quality_score(text)
    return result['overall_score'] < threshold


def analyze_text_sample(text: str, max_display: int = 100) -> None:
    """
    Debug helper: Print detailed quality analysis of a text sample.

    Args:
        text: Input text to analyze
        max_display: Maximum characters to display
    """
    result = calculate_text_quality_score(text)

    print(f"Text sample: {text[:max_display]}{'...' if len(text) > max_display else ''}")
    print(f"Length: {len(text)}")
    print(f"Overall score: {result['overall_score']:.3f}")
    print(f"Is valid: {result['is_valid']}")
    print(f"Issues: {result['issues']}")
    print("Metrics:")
    for key, value in result['metrics'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    # Test with known garbage
    garbage_samples = [
        "-duSs ie-DsichargeSdhNeoltvAiTtnogtLaqolucse",
        "IdU -Susie-1-ShareCalqluSewlriiytap",
        "Ccu AksMe PdMUUApSIE",
        "xdc lCuadmep us"
    ]

    # Test with known good text
    good_samples = [
        "The committee meeting was held on Tuesday at 2pm.",
        "Library staff discussed the new collection policy.",
        "John Smith presented the annual budget report."
    ]

    print("=== Garbage Samples ===")
    for sample in garbage_samples:
        result = calculate_text_quality_score(sample)
        print(f"'{sample[:40]}...': score={result['overall_score']:.2f}, valid={result['is_valid']}, issues={result['issues']}")

    print("\n=== Good Samples ===")
    for sample in good_samples:
        result = calculate_text_quality_score(sample)
        print(f"'{sample[:40]}...': score={result['overall_score']:.2f}, valid={result['is_valid']}, issues={result['issues']}")
