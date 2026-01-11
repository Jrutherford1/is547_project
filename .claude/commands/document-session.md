---
description: Document what happened this session for provenance tracking
---

Record this work session for project history and provenance.

## Your Task

1. **Gather Context**
   - Read recent git diff or changed files (use `Bash: git diff --name-only HEAD~5` or similar)
   - Read any updated docs in `docs/` (including `docs/code-reviews/`, `docs/features/`, and `docs/sessions/`)
   - Note what was discussed/attempted this session

2. **Create Session Document**

Create a new session file at `docs/sessions/session-YYYY-MM-DD-HHMM.md`:

```markdown
# Work Session: {YYYY-MM-DD HH:MM}

## Goals
- What the user wanted to accomplish

## What Happened
- Actions taken
- Files modified: `list of files`
- Agents/commands used

## Outcomes
- What succeeded
- What failed or was deferred
- Decisions made

## Open Items
- Unfinished work
- Questions to resolve
- Next steps

## Artifacts Produced
- Link to any new docs, plans, or outputs (e.g., code reviews in `docs/code-reviews/`, feature plans in `docs/features/`)
```

**Filename format**: Use `session-YYYY-MM-DD-HHMM.md` (e.g., `session-2026-01-07-1430.md`) to allow multiple sessions per day.

3. **Update CLAUDE.md if Needed**

If any new conventions, patterns, or project decisions were established, add them to `CLAUDE.md`.

4. **Create ADR if Significant Decision Made**

If a meaningful architectural or design decision was made, create:

`docs/decisions/ADR-{NNN}-{short-title}.md`

```markdown
# ADR-{NNN}: {Title}

**Date:** {date}
**Status:** Accepted

## Context
Why this decision was needed.

## Decision
What we chose to do.

## Consequences
- Positive outcomes
- Trade-offs accepted
- Follow-up work needed
```

## Output

Confirm what was documented and where.
