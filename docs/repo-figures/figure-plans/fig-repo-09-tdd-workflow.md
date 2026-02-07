# fig-repo-09: Test-Driven Development Workflow

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-09 |
| **Title** | Test-Driven Development: Write Tests First |
| **Complexity Level** | L3 (Technical practice) |
| **Target Persona** | Research Scientist, ML Engineer |
| **Location** | CONTRIBUTING.md, docs/ |
| **Priority** | P3 (Medium) |

## Purpose

Introduce TDD to researchers who may write tests as an afterthought, showing why writing tests first leads to better code.

## Key Message

"Write the test first, watch it fail, then write just enough code to make it pass. Repeat."

## Visual Concept

**Red-Green-Refactor cycle:**

```
┌─────────────────────────────────────────────────────────────────┐
│                  TDD: THE RED-GREEN-REFACTOR CYCLE              │
│                                                                 │
│                        ┌─────────┐                              │
│                   ┌───▶│  RED    │◀───┐                         │
│                   │    │ (Write  │    │                         │
│                   │    │ failing │    │                         │
│                   │    │  test)  │    │                         │
│                   │    └────┬────┘    │                         │
│                   │         │         │                         │
│                   │         ▼         │                         │
│              ┌────┴───┐         ┌────┴────┐                    │
│              │REFACTOR│         │  GREEN  │                    │
│              │(Clean  │◀────────│  (Make  │                    │
│              │ up)    │         │  it     │                    │
│              │        │         │  pass)  │                    │
│              └────────┘         └─────────┘                    │
│                                                                 │
│  EXAMPLE FROM THIS REPO:                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 1. RED: def test_ged_handles_different_mask_lengths():  │   │
│  │         → AssertionError (GED not implemented yet)      │   │
│  │                                                          │   │
│  │ 2. GREEN: Implement weighted covariance in ged.py       │   │
│  │         → Test passes!                                   │   │
│  │                                                          │   │
│  │ 3. REFACTOR: Clean up, add docstrings                   │   │
│  │         → Code is clean AND correct                      │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Content Elements

### Required Elements
1. Red-Green-Refactor circular diagram
2. Color coding (red, green, blue/gray)
3. Concrete example from this repository
4. Test name and outcome at each stage

### Optional Elements
1. Benefits list
2. Common mistakes
3. Link to pytest documentation

## Text Content

### Title Text
"TDD: Test First, Code Second"

### Labels/Annotations
- RED: "Write a failing test for desired behavior"
- GREEN: "Write minimum code to pass the test"
- REFACTOR: "Clean up while keeping tests green"
- Example: "Real example: GED decomposition fix"

### Caption (for embedding)
Test-Driven Development follows the Red-Green-Refactor cycle: write a failing test, make it pass with minimal code, then clean up. This ensures code works before it's written.

## Prompts for Nano Banana Pro

### Style Prompt
Circular workflow diagram with traffic light colors (red, green, blue/gray for refactor). Clean, professional. Include a code example panel at the bottom. Developer-friendly aesthetic.

### Content Prompt
Create a TDD cycle diagram:
1. Three connected nodes in a cycle: RED → GREEN → REFACTOR → (back to RED)
2. Each node has a short description
3. Below: A panel showing a concrete example with test code snippet

Colors: Red for failing tests, Green for passing, Blue/Gray for refactoring

### Refinement Notes
- The cycle nature should be visually clear
- Include actual test function names from this repo
- Show that refactoring happens AFTER the test passes

## Alt Text

Circular diagram showing the TDD cycle: Red (write failing test), Green (make it pass), Refactor (clean up), with a code example from the GED decomposition implementation.

## Status

- [x] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
