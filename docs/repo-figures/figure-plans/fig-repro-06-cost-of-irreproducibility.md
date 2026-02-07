# fig-repro-06: The Cost of Irreproducibility

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repro-06 |
| **Title** | The Cost of Irreproducibility |
| **Complexity Level** | L1 (ELI5) |
| **Target Persona** | PI, Grant Reviewer, Lab Manager |
| **Location** | README.md |
| **Priority** | P2 |
| **Aspect Ratio** | 16:10 |

## Purpose

Quantify the economic and scientific cost of irreproducibility to motivate investment in reproducibility infrastructure.

## Key Message

"Irreproducibility costs $28 billion annually in preclinical research alone. Proper tooling takes hours to set up but saves months of wasted effort."

## Literature Sources

| Source | Finding | DOI/URL |
|--------|---------|---------|
| Freedman et al. 2015 | $28B annual waste in preclinical research | [10.1371/journal.pbio.1002165](https://doi.org/10.1371/journal.pbio.1002165) |
| Baker 2016 (Nature) | 70% of researchers failed to reproduce others' experiments | [10.1038/533452a](https://doi.org/10.1038/533452a) |

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    THE COST OF IRREPRODUCIBILITY                                │
│                    When science can't be verified                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ANNUAL COST: PRECLINICAL RESEARCH (USA)                                        │
│  ═══════════════════════════════════════                                        │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │                     $28 BILLION / YEAR                                  │   │
│  │                     ══════════════════                                  │   │
│  │                                                                         │   │
│  │    Wasted on studies that cannot be replicated                          │   │
│  │    (Freedman et al. 2015)                                               │   │
│  │                                                                         │   │
│  │    Breakdown:                                                           │   │
│  │    ├── Study design/methods    $5.0B                                    │   │
│  │    ├── Biological reagents     $4.0B                                    │   │
│  │    ├── Reference materials     $3.5B                                    │   │
│  │    ├── Lab protocols           $3.0B                                    │   │
│  │    └── Data analysis/reporting $12.5B   ← Software reproducibility!    │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  INDIVIDUAL RESEARCHER COSTS                                                    │
│  ═══════════════════════════                                                    │
│                                                                                 │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐           │
│  │                   │  │                   │  │                   │           │
│  │  3-6 MONTHS       │  │  $50-200K         │  │  2+ YEARS         │           │
│  │  ───────────      │  │  ───────────      │  │  ───────────      │           │
│  │  Time lost trying │  │  Grant money on   │  │  Career delay     │           │
│  │  to reproduce own │  │  failed follow-up │  │  from retraction  │           │
│  │  old work         │  │  studies          │  │  or correction    │           │
│  │                   │  │                   │  │                   │           │
│  └───────────────────┘  └───────────────────┘  └───────────────────┘           │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  THE ROI OF REPRODUCIBILITY TOOLING                                             │
│  ═══════════════════════════════════                                            │
│                                                                                 │
│  Setup cost (Foundation PLR approach):                                          │
│  • uv + lockfile: 30 minutes                                                    │
│  • DuckDB migration: 2-4 hours                                                  │
│  • MLflow integration: 4-8 hours                                                │
│  • Documentation: 2-4 hours                                                     │
│  ─────────────────────────────                                                  │
│  Total: ~1-2 days of effort                                                     │
│                                                                                 │
│  Return:                                                                        │
│  • Never lose 3-6 months to broken code                                         │
│  • Reviews verify in 8 minutes instead of "trust me"                            │
│  • Future students can build on your work                                       │
│                                                                                 │
│  ROI: 1-2 days vs potentially months/years → 100-1000x return                   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **$28B headline**: Large, prominent figure
2. **Breakdown pie/bar**: Where the money goes
3. **Three individual cost cards**: Time, money, career
4. **ROI calculation**: Setup time vs return

## Text Content

### Title Text
"The Cost of Irreproducibility: $28 Billion and Counting"

### Caption
Irreproducibility costs US preclinical research $28 billion annually, with $12.5B attributable to data analysis and reporting (Freedman et al. 2015, [DOI](https://doi.org/10.1371/journal.pbio.1002165)). Individual researchers lose 3-6 months per failed reproduction attempt. Foundation PLR's reproducibility tooling takes 1-2 days to set up but prevents these losses—a 100-1000x return on investment.

## Prompts for Nano Banana Pro

### Style Prompt
Large $28B headline. Breakdown bar chart showing where money is lost. Three cost cards for individual impact. ROI section showing investment vs return. Professional, sobering color palette—communicate the seriousness without being alarmist.

### Content Prompt
Create "Cost of Irreproducibility" infographic:

**TOP - Big Number**:
- $28 BILLION / YEAR
- Breakdown bar showing data analysis as largest component

**MIDDLE - Three Cards**:
- Time: 3-6 months lost
- Money: $50-200K wasted
- Career: 2+ year delays

**BOTTOM - ROI**:
- Setup: 1-2 days
- Return: Avoid months/years of waste
- ROI: 100-1000x

## Alt Text

Infographic on irreproducibility costs. Headline: $28 billion annually wasted in US preclinical research (Freedman 2015). Breakdown shows data analysis/reporting as largest component at $12.5B. Individual researcher costs: 3-6 months lost reproducing own work, $50-200K grant money wasted, 2+ year career delays from retractions. ROI of reproducibility tooling: 1-2 days setup yields 100-1000x return by preventing these losses.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in README.md

