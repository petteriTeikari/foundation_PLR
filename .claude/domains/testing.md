# Testing Domain Context

## Quick Reference

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific module
pytest tests/unit/test_stats.py
```

## Test Organization

```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Component integration tests
├── smoke/          # Quick validation tests
└── conftest.py     # Shared fixtures
```

## Test Categories

| Category | Location | Purpose | Duration |
|----------|----------|---------|----------|
| Unit | `tests/unit/` | Individual function tests | <1s each |
| Integration | `tests/integration/` | End-to-end stage tests | <30s each |
| Smoke | `tests/smoke/` | Critical path validation | <5s each |

## Key Fixtures (conftest.py)

- `sample_plr_signal` - Sample PLR time series
- `sample_predictions` - y_true, y_prob for metric tests
- `sample_config` - Minimal Hydra config

## Docker Tests

Some tests require Docker for R environments:

```bash
# Run Docker tests
pytest tests/ -m docker

# Skip Docker tests
pytest tests/ -m "not docker"
```

## Excluded from Tests

Vendored code is excluded:
- `src/imputation/pypots/`
- `src/imputation/nuwats/`
- `src/classification/tabpfn/`
- `src/classification/tabpfn_v1/`

## Coverage Targets

| Module | Target |
|--------|--------|
| `src/stats/` | 90%+ |
| `src/classification/` | 80%+ |
| `src/viz/` | 70%+ |

## Writing Tests

NumPy-style docstrings in tests:

```python
def test_compute_auroc_perfect():
    """
    Test AUROC with perfect discrimination.

    Given: Perfect predictions
    When: AUROC computed
    Then: Should equal 1.0
    """
    ...
```

## CI Integration

Tests run on every PR. Ensure:
1. All tests pass locally
2. No new warnings introduced
3. Coverage doesn't decrease
