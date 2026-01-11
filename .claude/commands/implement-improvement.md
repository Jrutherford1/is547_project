---
description: Implement a specific improvement from the architectural review
---

Implement one improvement from `docs/architectural-review.md`, verify it, and document the change.

## Input

User specifies which improvement to tackle, e.g.:
```
/implement-improvement Extract Search Logic from API Module
```

## Process

### Step 1: Understand the Improvement
- Read the latest code review from `docs/code-reviews/` (e.g., `code-review-YYYY-MM-DD.md`)
- Find the specified improvement
- Understand what needs to change and why

### Step 2: Plan the Change
Before coding, outline:
- Files to modify
- Files to create
- Expected impact on other code
- Any risks

### Step 3: Implement

**Coding Standards (Python):**
- Follow PEP 8
- Include type hints on all functions
- Write docstrings for public functions (Google style)
- Use specific exception types
- Use `pathlib` for file paths
- Use f-strings for formatting
- Handle edge cases (None, empty, boundary values)

**Implementation Approach:**
- Make minimal, focused changes
- Preserve existing behavior
- Don't refactor unrelated code
- Add comments explaining "why" for complex logic

### Step 4: Verify
- Run existing tests: `Bash: pytest` (if available)
- Check for syntax errors: `Bash: python -m py_compile {file}`
- Verify imports work: `Bash: python -c "from api.main import app"`

### Step 5: Document

Update the code review file in `docs/code-reviews/`:
- Mark the item as ✅ **DONE** with date
- Note any deviations from the original plan

Append to `docs/changelog.md` (create if needed in the docs root):
```markdown
## {YYYY-MM-DD}

### {Improvement Title}
- **What:** Brief description of change
- **Why:** Reference to architectural review
- **Files changed:** List of files
- **Testing:** How it was verified
```

## Output

Provide a summary:
1. What was implemented
2. Files changed
3. Test results
4. Any follow-up work needed
