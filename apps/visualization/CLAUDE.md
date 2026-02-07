# Foundation PLR Visualization - Development Guidelines

## 🚨 CRITICAL: Registry is Single Source of Truth

**ALL method names come from `configs/mlflow_registry/` - NEVER hardcode!**

| Parameter | Count | Registry Location |
|-----------|-------|-------------------|
| Outlier methods | **11** | `parameters/classification.yaml` |
| Imputation methods | **8** | `parameters/classification.yaml` |
| Classifiers | **5** | `parameters/classification.yaml` |

**If visualization shows different counts (e.g., 17 outlier methods), the DATA IS WRONG.**

See: `.claude/rules/05-registry-source-of-truth.md`

---

## Core Principles

### 1. Content/Style Separation (CRITICAL)

**All visual styling MUST be defined in CSS, never hard-coded in components.**

```tsx
// BAD - Hard-coded styles
<path stroke="#0077BB" strokeWidth={2} />
<text fontSize={12} fontFamily="Manrope" />

// GOOD - CSS custom properties
<path className="data-line" />
<text className="axis-label" />
```

**Rationale:**
- Single source of truth for all visual decisions
- Easy global style updates without touching components
- Consistent look across all figures
- Print styles can override via CSS media queries

### 2. Data-Driven Labels (CRITICAL)

**All text labels that depend on data MUST be derived from data fields, never hard-coded.**

```tsx
// BAD - Hard-coded metric name
<text>AUROC vs Retention Rate</text>
title="AUROC vs Retention Rate"

// GOOD - Derived from data
<text>{METRIC_LABELS[data.metric]} vs Retention Rate</text>
title={`${METRIC_LABELS[data.metric]} vs Retention Rate`}
```

**Domain-specific terms that should NEVER be hard-coded in components:**
- AUROC, AUC, Brier Score (use `data.metric`)
- Glaucoma, Control, Case (use `data.case_label`, `data.control_label`)
- Any specific threshold values (use `data.threshold`)

**Rationale:**
- Components are reusable across different metrics/domains
- User of the component provides the domain context via data
- Same component works for AUROC, Brier, Net Benefit, etc.

### 2. Shared Style System

All figures share `src/styles/foundations.css` which defines:
- Typography (fonts, sizes, weights)
- Colors (Paul Tol colorblind-safe palette)
- Spacing and layout
- Line weights and strokes
- Interactive states

**Never create figure-specific CSS files.** If a figure needs unique styling:
1. First, ask if it truly needs to be different
2. If yes, add a CSS class to `foundations.css`
3. Document why the exception exists

### 3. D3.js Pattern: Calculate, Don't Render

D3 is for **mathematical calculations only**:
- Scales (`d3.scaleLinear()`)
- Line/area generators (`d3.line()`)
- Statistical functions

React renders all SVG elements. Never use:
- `d3.select().append()`
- `d3.select().attr()`
- Direct DOM manipulation via refs

### 4. Component Structure

Every figure component should:
1. Accept data via props (validated with Zod)
2. Use `useMemo` for all D3 calculations
3. Return pure SVG with CSS classes
4. Support both static and interactive modes

### 5. Typography Stack

```
Headings:  Neue Haas Grotesk Display Black (local)
Labels:    Manrope Medium
Values:    JetBrains Mono Regular
```

Never override fonts inline. Use:
- `.figure-title` for headings
- `.axis-label` for axis labels
- `.tick-label` for tick values
- `.annotation` for callouts

### 6. Color Usage

Always use CSS custom properties:
```css
var(--color-primary)      /* Main data */
var(--color-secondary)    /* Comparisons */
var(--color-baseline)     /* Reference lines */
var(--color-ci)           /* Confidence intervals */
```

### 7. Export Requirements

All figures must support:
- SVG export (text preserved)
- PDF export (via svg2pdf.js)
- PNG export (high DPI for print)

Static exports should look identical to interactive versions.

## File Organization

```
visualization/
├── src/
│   ├── components/
│   │   └── figures/       # All figure components
│   ├── styles/
│   │   └── foundations.css  # THE ONE STYLE FILE
│   ├── types/             # TypeScript + Zod schemas
│   ├── utils/             # D3 helpers, export functions
│   └── hooks/             # React hooks
├── CLAUDE.md              # This file
└── package.json
```

## Adding a New Figure

1. Create component in `src/components/figures/`
2. Use existing CSS classes from `foundations.css`
3. Add Zod schema to `src/types/figures.ts`
4. Write tests for data validation
5. Do NOT add figure-specific CSS

## Checklist Before Commit

- [ ] No hard-coded colors, fonts, or sizes in TSX
- [ ] All visual classes defined in `foundations.css`
- [ ] Zod schema validates input data
- [ ] Component works in both static and interactive modes
- [ ] Export to SVG/PDF tested
