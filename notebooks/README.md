# Notebooks (`notebooks/`)

Jupyter notebooks for tutorials and interactive exploration.

## Available Notebooks

| Notebook | Purpose |
|----------|---------|
| `comprehensive_guide.ipynb` | Complete guide to the Foundation PLR pipeline |
| `data_access_tutorial.ipynb` | How to access and work with PLR data |
| `reproducibility_tutorial.ipynb` | Reproducing experiment results |

## Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Start Jupyter
jupyter lab notebooks/
```

## Notebook Descriptions

### 1. `comprehensive_guide.ipynb`

**Audience**: New users, researchers

**Contents**:
- Project overview and research question
- Pipeline walkthrough (outlier → imputation → features → classification)
- STRATOS metrics explanation
- Example analysis workflow

### 2. `data_access_tutorial.ipynb`

**Audience**: Data scientists, developers

**Contents**:
- Connecting to DuckDB database
- Loading PLR signals
- Understanding data schema
- Working with ground truth masks
- Subject counts and stratification

### 3. `reproducibility_tutorial.ipynb`

**Audience**: Researchers wanting to reproduce results

**Contents**:
- Environment setup
- Loading MLflow experiments
- Recreating figures
- Statistical analysis reproduction
- Comparison with published results

## Running Notebooks

### Prerequisites

```bash
# Install Jupyter (if not already)
uv pip install jupyterlab

# Activate environment
source .venv/bin/activate
```

### Launch

```bash
# JupyterLab (recommended)
jupyter lab notebooks/

# Classic Jupyter
jupyter notebook notebooks/
```

## Data Access in Notebooks

```python
import duckdb

# Connect to database
conn = duckdb.connect('../SERI_PLR_GLAUCOMA.db', read_only=True)

# Query subjects
df = conn.execute("SELECT * FROM train LIMIT 10").fetchdf()
```

## Best Practices

1. **Don't commit output cells** - Clear outputs before committing
2. **Use relative paths** - For portability
3. **Document prerequisites** - List required packages at top
4. **Include expected outputs** - Show what results should look like

## Converting to Python Scripts

```bash
jupyter nbconvert --to script notebooks/data_access_tutorial.ipynb
```

## Adding New Notebooks

1. Create notebook in `notebooks/`
2. Add description to this README
3. Test with clean kernel restart
4. Clear outputs before committing
