# /update-readmes - Repository Documentation Auto-Update

Update all README.md files, audit docstrings, and plan Nano Banana Pro figures.

Read the skill definition at `.claude/skills/update-readmes/SKILL.md` and follow the Ralph Wiggum pattern:

1. **Agent 1 (Discovery)**: Read `.claude/skills/update-readmes/protocols/agent-1-discovery.md` and scan for stale/broken documentation
2. **Agent 2 (Generation)**: Read `.claude/skills/update-readmes/protocols/agent-2-generation.md` and generate/update content
3. **Human Approval**: Present categorized changes for sign-off
4. **Git Checkpoint**: Commit after each approved category
5. **Validate**: Verify all cross-references resolve
6. **Iterate**: Repeat until convergence (max 3 iterations)

## Scope Options

Parse the argument after `/update-readmes`:
- No argument or `full` = all three tasks (readmes + docstrings + figures)
- `report` = discovery only, no edits
- `scope=readmes` = only README.md updates
- `scope=docstrings` = only docstring audit/additions
- `scope=figures` = only Nano Banana Pro figure plans

## State Tracking

Load/save progress in `.claude/skills/update-readmes/state/update-state.json`

## Reference Files

- `.claude/skills/update-readmes/reference/readme-template.md` - Standard README structure
- `.claude/skills/update-readmes/reference/docstring-standards.md` - NumPy docstring format
- `.claude/skills/update-readmes/reference/figure-plan-checklist.yaml` - Figure plan validation
