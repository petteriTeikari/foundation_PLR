# Agent 2: Clarity & Architecture

## Mandate

Ensure optimized content is maximally effective for LLM instruction-following.

## Input

- Agent 1's compression proposals
- The target files being modified

## Review Criteria

### 1. Did Compression Lose Critical Nuance?

For each proposal, check:
- Does the compressed version preserve the BEHAVIORAL intent?
- Would an LLM reading only the compressed version follow the rule?
- Are edge cases still covered?

### 2. Front-Load Critical Rules

LLMs weight early tokens more heavily. Restructure files so:
- First 25% = most critical rules (MUST/NEVER/BANNED)
- Middle 50% = standard procedures (HOW-TO)
- Last 25% = reference material (tables, examples, cross-refs)

### 3. Hierarchy Placement

Verify content is at the correct level:

| Level | Content | Tone |
|-------|---------|------|
| Root CLAUDE.md | WHAT the project is, WHY rules exist | Concise overview |
| .claude/CLAUDE.md | HOW to behave, specific patterns | Imperative contract |
| .claude/rules/ | MUST-follow rules (one per file) | Binary MUST/NEVER |
| .claude/domains/ | WHEN-NEEDED context | Reference material |

### 4. Imperative Over Descriptive

Reword for LLM effectiveness:

| Weak (descriptive) | Strong (imperative) |
|--------------------|---------------------|
| "You should consider using" | "USE" |
| "It is recommended to" | "MUST" |
| "Try to avoid" | "NEVER" |
| "It would be best if" | "ALWAYS" |

### 5. Cross-Reference Clarity

When content was moved, verify:
- The cross-reference is findable (exact path)
- The reference clearly states WHAT was moved
- No circular references

## Output Format

```yaml
reviews:
  - proposal_id: C01
    verdict: "accept" | "revise" | "reject"
    clarity_score: 8  # 1-10
    issues:
      - "Compressed table lost the BANNED code example"
    revised_content: "..."  # Only if verdict is 'revise'
```

## Constraints

- Clarity > compression when in conflict
- Every file must start with its most important rule
- Cross-references must use exact file paths
- Tables for rules, prose only when context is essential
