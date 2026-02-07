# Installation

## Requirements

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | ≥3.11 | Required for modern type hints |
| UV | Latest | Package manager (conda banned) |
| Git | Latest | Version control |
| R | ≥4.4 | Optional, for pminternal stability analysis |

## Step 1: Clone the Repository

```bash
git clone https://github.com/petteriTeikari/foundation_PLR.git
cd foundation-PLR/foundation_PLR
```

## Step 2: Install Dependencies

!!! warning "UV Only"
    This project uses UV for package management. **Conda and pip are not supported.**

```bash
# Install UV if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

## Step 3: Verify Installation

```bash
# Activate the virtual environment
source .venv/bin/activate

# Verify imports work
python -c "from src.data_io import data_import; print('Installation successful!')"
```

## Optional: R Setup for pminternal

For STRATOS-compliant model stability analysis:

```bash
# Install R (Ubuntu/Debian)
sudo apt-get install r-base

# Install pminternal package (in R)
Rscript -e "install.packages('pminternal')"
```

## Troubleshooting

### UV not found

```bash
# Add UV to PATH
export PATH="$HOME/.cargo/bin:$PATH"
```

### Missing system dependencies

```bash
# Ubuntu/Debian
sudo apt-get install build-essential python3-dev
```
