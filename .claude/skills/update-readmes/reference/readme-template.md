# README.md Template

Standard structure for directory-level READMEs in this project.

## Root README.md

```markdown
# Foundation PLR

> Research question summary

## 30-Second Summary
Key findings (4-5 bullet points)

## Quick Navigation
| I want to... | Go to |
|--------------|-------|
| Task | Link |

## Quick Start
Setup instructions (Docker or manual)

## Running the Pipeline
Make commands

## Project Structure
Directory tree with descriptions

## Testing
How to run tests

## Contributing
Link to CONTRIBUTING.md

## License
MIT
```

## Source Module README.md (`src/*/README.md`)

```markdown
# Module Name

One-line purpose.

## Overview
2-3 sentence description of what this module does and why it exists.

## Modules

| Module | Purpose |
|--------|---------|
| `file.py` | What it does |

## Key Functions

| Function | Module | Description |
|----------|--------|-------------|
| `func()` | `file.py` | What it does |

## Usage

\```python
from src.module import function
result = function(args)
\```

## Configuration

| Config File | Purpose |
|-------------|---------|
| `configs/X.yaml` | What it configures |

## See Also
- Related module links
```

## Config Directory README.md (`configs/*/README.md`)

```markdown
# Config Category Name

What these configs control.

## Files

| File | Purpose |
|------|---------|
| `file.yaml` | What it configures |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `param` | type | value | What it does |

## Hydra Usage

How to override via CLI:
\```bash
python -m src.module param=value
\```

## See Also
- Link to source module that loads these configs
```

## Progressive Disclosure Checklist

Every directory README should support:

- [ ] Level 1: Link to relevant infographic (if exists)
- [ ] Level 2: Full parameter/module documentation
- [ ] Level 3: Links to actual YAML/source for deep dive
- [ ] Level 4: Links to external references (papers, docs)
