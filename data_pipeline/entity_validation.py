"""
Entity Validation Module

Provides validation functions for NLP-extracted entities.
Filters out garbage, false positives, and malformed entities.
"""

import re
import unicodedata
from typing import Tuple, Optional, Set

# Expanded filter terms for PERSON entities
# These are terms that spaCy often incorrectly classifies as people
PERSON_FILTER_TERMS: Set[str] = {
    # Generic roles (from original list)
    "Library", "Librarian", "Librarians", "Staff", "Committee",
    "Chair", "Chairperson", "Member", "Members", "Director",
    "University", "UIUC", "Illinois", "Department",

    # Meeting/document terms
    "Absent", "Present", "Guest", "Guests", "Visitor", "Visitors",
    "Note", "Notes", "Action", "Item", "Items", "Update", "Updates",
    "Report", "Reports", "Review", "Discussion", "Motion", "Vote",
    "Agenda", "Minutes", "Attendee", "Attendees",

    # Generic titles without names
    "Dean", "Associate Dean", "Assistant Dean", "Vice", "Deputy",
    "Manager", "Coordinator", "Specialist", "Analyst", "Administrator",
    "Head", "Lead", "Senior", "Junior", "Interim",

    # Software/system names commonly misclassified
    "Alma", "Aeon", "Archon", "Voyager", "Espresso", "PaperCut",
    "VuFind", "Primo", "Illiad", "OCLC", "WorldCat", "CARLI",
    "Outlook", "Teams", "Zoom", "WebEx", "SharePoint", "Box",

    # Building/location names that aren't people
    "Oak Street", "Krannert", "Grainger", "Main Library", "ACES",
    "Funk", "Ricker", "UGL", "Undergraduate Library",
    "Main Stacks", "Bookstacks", "SSHEL", "CPLA",

    # Common false positives from analysis
    "Training", "Info", "Information", "Maps", "Space", "Spaces",
    "Service", "Services", "Access", "Collection", "Collections",
    "Digital", "Public", "Special", "Reference", "Circulation",
    "Preservation", "Conservation", "Acquisitions", "Cataloging",

    # Measurement/quantity terms
    "cubic feet", "linear feet", "boxes", "items", "volumes",

    # Time-related false positives
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",

    # Generic terms
    "Group", "Team", "Unit", "Office", "Council", "Board",
    "Faculty", "Students", "Graduate", "Undergraduate",
    "Project", "Program", "Initiative", "Task Force",
}

# Lowercase version for case-insensitive matching
PERSON_FILTER_TERMS_LOWER = {term.lower() for term in PERSON_FILTER_TERMS}

# Patterns that indicate invalid PERSON entities
PERSON_INVALID_PATTERNS = [
    r'\n',                    # Contains newline
    r'\t',                    # Contains tab
    r' - [A-Z]',              # "Name - Sentence fragment" pattern
    r' – [A-Z]',              # Em-dash version
    r'^\d',                   # Starts with digit
    r'\d{4}',                 # Contains 4-digit year
    r'^[a-z]',                # Starts with lowercase
    r'[<>{}|\[\]]',           # Contains brackets/special chars
    r'^\W',                   # Starts with non-word character
    r'\W$',                   # Ends with non-word character (except period)
    r'\\',                    # Contains backslash
    r'@',                     # Contains @ symbol
    r'https?:',               # Contains URL
    r'\.com|\.org|\.edu',     # Contains domain
    r'\s{2,}',                # Multiple consecutive spaces
]

# Compiled patterns for efficiency
PERSON_INVALID_COMPILED = [re.compile(p) for p in PERSON_INVALID_PATTERNS]

# Valid characters for person names
VALID_NAME_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ -'.áéíóúñüöäàèìòùâêîôûçÁÉÍÓÚÑÜÖÄÀÈÌÒÙÂÊÎÔÛÇ")

VOWELS = set('aeiouAEIOU')


def clean_entity_text(entity: str) -> str:
    """
    Clean entity text by normalizing whitespace and unicode.

    Args:
        entity: Raw entity text

    Returns:
        Cleaned entity text
    """
    if not entity:
        return ""

    # Normalize unicode
    text = unicodedata.normalize('NFKC', entity)

    # Replace various whitespace with single space
    text = re.sub(r'[\n\r\t]+', ' ', text)

    # Collapse multiple spaces
    text = re.sub(r' +', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def calculate_name_vowel_ratio(name: str) -> float:
    """
    Calculate vowel ratio for a name.

    Real names typically have 30-50% vowels.
    Gibberish often has extreme ratios.

    Args:
        name: Person name to check

    Returns:
        Vowel ratio (0.0 to 1.0)
    """
    letters = [c for c in name if c.isalpha()]
    if not letters:
        return 0.0
    vowel_count = sum(1 for c in letters if c in VOWELS)
    return vowel_count / len(letters)


def is_plausible_name(text: str) -> Tuple[bool, str]:
    """
    Check if text could plausibly be a person's name.

    Args:
        text: Text to validate

    Returns:
        Tuple of (is_valid, reason_if_invalid)
    """
    if not text:
        return False, "empty"

    # Length check: names are typically 2-60 characters
    if len(text) < 2:
        return False, "too_short"
    if len(text) > 60:
        return False, "too_long"

    # Word count: names typically have 1-5 words
    words = text.split()
    if len(words) > 5:
        return False, "too_many_words"

    # Check for invalid characters
    invalid_chars = set(text) - VALID_NAME_CHARS - {' '}
    if invalid_chars:
        return False, f"invalid_chars:{repr(invalid_chars)}"

    # Each word should start with capital letter (for Western names)
    # Allow some flexibility for prefixes like "de", "von", "van"
    lowercase_prefixes = {'de', 'la', 'le', 'von', 'van', 'der', 'del', 'di', 'da'}
    for i, word in enumerate(words):
        if not word:
            continue
        # First word must be capitalized
        if i == 0 and word[0].islower():
            return False, "first_word_lowercase"
        # Other words can be lowercase if they're common prefixes
        if i > 0 and word[0].islower() and word.lower() not in lowercase_prefixes:
            return False, "word_lowercase"

    # Check vowel ratio per word
    for word in words:
        if len(word) > 2:  # Only check longer words
            ratio = calculate_name_vowel_ratio(word)
            if ratio < 0.15:  # Almost no vowels - likely gibberish
                return False, "low_vowel_ratio"
            if ratio > 0.75:  # Almost all vowels - unusual
                return False, "high_vowel_ratio"

    # Check overall vowel ratio for the full name
    overall_vowel_ratio = calculate_name_vowel_ratio(text)
    if overall_vowel_ratio < 0.20:  # Names typically have 25-45% vowels
        return False, "overall_low_vowel_ratio"

    # Check for too many uppercase letters (sign of OCR/encoding issues)
    letters = [c for c in text if c.isalpha()]
    if letters:
        upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if upper_ratio > 0.6 and len(text) > 5:  # More than 60% uppercase is suspicious
            return False, "too_many_uppercase"

    # Check for repeated character patterns (sign of OCR garbage)
    for word in words:
        if len(word) >= 4:
            # Check for 3+ consecutive same characters
            if re.search(r'(.)\1{2,}', word):
                return False, "repeated_chars"

    return True, "valid"


def validate_person_entity(entity: str) -> Tuple[bool, str]:
    """
    Validate a PERSON entity.

    Applies multiple validation rules to filter garbage and false positives.

    Args:
        entity: Entity text to validate

    Returns:
        Tuple of (is_valid, reason_if_invalid)
    """
    # Clean the entity first
    cleaned = clean_entity_text(entity)

    if not cleaned:
        return False, "empty_after_clean"

    # Check against filter terms
    if cleaned in PERSON_FILTER_TERMS:
        return False, "filter_term_exact"

    cleaned_lower = cleaned.lower()
    if cleaned_lower in PERSON_FILTER_TERMS_LOWER:
        return False, "filter_term"

    # Check if any filter term is contained in the entity
    for term in PERSON_FILTER_TERMS_LOWER:
        if len(term) > 3 and term in cleaned_lower:
            return False, f"contains_filter_term:{term}"

    # Check against invalid patterns
    for i, pattern in enumerate(PERSON_INVALID_COMPILED):
        if pattern.search(entity):  # Check original, not cleaned
            return False, f"pattern:{PERSON_INVALID_PATTERNS[i]}"

    # Check if it's a plausible name
    is_plausible, reason = is_plausible_name(cleaned)
    if not is_plausible:
        return False, f"not_plausible_name:{reason}"

    return True, "valid"


def validate_org_entity(entity: str) -> Tuple[bool, str]:
    """
    Validate an ORG entity.

    Args:
        entity: Entity text to validate

    Returns:
        Tuple of (is_valid, reason_if_invalid)
    """
    cleaned = clean_entity_text(entity)

    if not cleaned:
        return False, "empty_after_clean"

    # Length check
    if len(cleaned) < 2:
        return False, "too_short"
    if len(cleaned) > 150:
        return False, "too_long"

    # Check for embedded newlines/tabs in original
    if '\n' in entity or '\t' in entity:
        return False, "contains_newline_or_tab"

    # Check vowel ratio (org names can be acronyms, so be lenient)
    letters = [c for c in cleaned if c.isalpha()]
    if letters:
        vowel_ratio = sum(1 for c in letters if c in VOWELS) / len(letters)
        # Very low vowel ratio for non-acronyms is suspicious
        if len(cleaned) > 10 and vowel_ratio < 0.10:
            return False, "low_vowel_ratio"

    return True, "valid"


def validate_gpe_entity(entity: str) -> Tuple[bool, str]:
    """
    Validate a GPE (geopolitical entity/location) entity.

    Args:
        entity: Entity text to validate

    Returns:
        Tuple of (is_valid, reason_if_invalid)
    """
    cleaned = clean_entity_text(entity)

    if not cleaned:
        return False, "empty_after_clean"

    # Length check
    if len(cleaned) < 2:
        return False, "too_short"
    if len(cleaned) > 100:
        return False, "too_long"

    # Check for newlines
    if '\n' in entity or '\t' in entity:
        return False, "contains_newline_or_tab"

    # GPE starting with ~ or numbers is suspicious (e.g., "~50")
    if cleaned.startswith('~') or cleaned[0].isdigit():
        return False, "starts_with_number_or_symbol"

    return True, "valid"


def validate_date_entity(entity: str) -> Tuple[bool, str]:
    """
    Validate a DATE entity.

    Args:
        entity: Entity text to validate

    Returns:
        Tuple of (is_valid, reason_if_invalid)
    """
    cleaned = clean_entity_text(entity)

    if not cleaned:
        return False, "empty_after_clean"

    # Length check
    if len(cleaned) < 2:
        return False, "too_short"
    if len(cleaned) > 100:
        return False, "too_long"

    # Check for newlines
    if '\n' in entity or '\t' in entity:
        return False, "contains_newline_or_tab"

    # Dates like "13,4I4 3" are garbage - check for reasonable format
    # Allow: "2023", "January 15", "next week", "2023-01-15", etc.
    # Reject: excessive commas, mixed letters/numbers in odd patterns

    # Count commas - dates shouldn't have more than 2
    if cleaned.count(',') > 2:
        return False, "too_many_commas"

    # Check for garbage patterns like "13,4I4 3"
    # These have uppercase I or O mixed with numbers (OCR confusion)
    if re.search(r'\d[IO]\d', cleaned):
        return False, "ocr_garbage_pattern"

    return True, "valid"


def validate_entity(entity: str, entity_type: str) -> Tuple[bool, str]:
    """
    Validate any entity based on its type.

    Args:
        entity: Entity text to validate
        entity_type: One of "PERSON", "ORG", "GPE", "DATE"

    Returns:
        Tuple of (is_valid, reason_if_invalid)
    """
    validators = {
        "PERSON": validate_person_entity,
        "ORG": validate_org_entity,
        "GPE": validate_gpe_entity,
        "DATE": validate_date_entity,
    }

    validator = validators.get(entity_type)
    if validator:
        return validator(entity)

    # Unknown entity type - accept by default
    return True, "unknown_type"


if __name__ == "__main__":
    # Test with known problem entities from the dataset
    test_cases = [
        # Garbage from the problem file
        ("PERSON", "-duSs ie-DsichargeSdhNeoltvAiTtnogtLaqolucse"),
        ("PERSON", "IdU -Susie-1-ShareCalqluSewlriiytap"),
        ("PERSON", "Ccu AksMe PdMUUApSIE"),
        ("PERSON", "xdc lCuadmep us"),

        # Entities with newlines
        ("PERSON", "Lynne Thomas\nAbsent"),
        ("PERSON", "Hannah Williams\nBox"),

        # Sentence fragments
        ("PERSON", "Tom - I"),
        ("PERSON", "Lynne - We"),

        # Valid names that should pass
        ("PERSON", "Tom Teper"),
        ("PERSON", "Lynne Thomas"),
        ("PERSON", "Mary Smith-Jones"),
        ("PERSON", "José García"),

        # Filter terms that should be rejected
        ("PERSON", "Library"),
        ("PERSON", "Committee Chair"),
        ("PERSON", "Alma"),

        # Garbage dates
        ("DATE", "13,4I4 3"),
        ("DATE", "20141 1"),

        # Valid dates
        ("DATE", "2023"),
        ("DATE", "January 15, 2024"),

        # Garbage GPE
        ("GPE", "~50"),
        ("GPE", "Maiilt"),

        # Valid GPE
        ("GPE", "Illinois"),
        ("GPE", "Urbana"),
    ]

    print("=== Entity Validation Tests ===\n")
    for entity_type, entity in test_cases:
        is_valid, reason = validate_entity(entity, entity_type)
        status = "PASS" if is_valid else "REJECT"
        print(f"[{status}] {entity_type}: '{entity[:40]}{'...' if len(entity) > 40 else ''}' -> {reason}")
