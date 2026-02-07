# Interactive Visualization (`visualization/`)

This directory contains a React + TypeScript + D3.js application for interactive figures.

## Quick Start

```bash
cd visualization

# Install dependencies
npm install

# Start development server
npm run dev
# Open http://localhost:5173
```

## Tech Stack

| Technology | Purpose |
|------------|---------|
| React 18 | UI framework |
| TypeScript | Type safety |
| D3.js | Data visualization |
| Vite | Build tool |

## Project Structure

```
visualization/
├── package.json           # Dependencies
├── package-lock.json
├── tsconfig.json          # TypeScript config
├── vite.config.ts         # Vite build config
├── index.html             # Entry HTML
│
├── src/                   # Source code
│   ├── components/        # React components
│   └── ...
│
├── scripts/               # Build scripts
│
└── node_modules/          # Dependencies (git-ignored)
```

## Available Scripts

```bash
# Development
npm run dev              # Start dev server

# Build
npm run build            # Build for production

# Preview
npm run preview          # Preview production build

# Lint
npm run lint             # Run ESLint
```

## Relationship to Python Figures

This directory provides **interactive** versions of manuscript figures, while `src/viz/` generates **static** figures:

| Use Case | Source |
|----------|--------|
| Manuscript (PDF/print) | `src/viz/` (Python + matplotlib) |
| Supplementary (web) | `visualization/` (React + D3) |
| Data exploration | `visualization/` (React + D3) |

## Data Flow

```
MLflow experiments → DuckDB → JSON → D3.js visualization
                              ↑
                   figures/generated/data/*.json
```

The React app reads JSON data exported by Python scripts:

```typescript
// Example: Loading figure data
const data = await fetch('/data/fig_retained_multi_metric.json');
```

## Building for Production

```bash
npm run build
# Output in dist/
```

## Development

See [visualization/CLAUDE.md](CLAUDE.md) for AI assistant context.

### Adding a New Component

1. Create component in `src/components/`
2. Import D3.js for data visualization
3. Use shared data from `figures/generated/data/`

### TypeScript Configuration

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "lib": ["ES2020", "DOM"],
    "strict": true
  }
}
```
