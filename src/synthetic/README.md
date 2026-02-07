# Synthetic

Synthetic PLR data generation for testing and public demos.

## Overview

Generates synthetic pupillary light reflex curves from parametric physiological models (not derived from real patient data). Used to create `SYNTH_PLR_DEMO.db` for testing the pipeline without access to private clinical data. Includes artifact injection for realistic signal corruption and privacy validation to ensure synthetic data is not overly similar to real data.

## Modules

| Module | Purpose |
|--------|---------|
| `demo_dataset.py` | Public API for generating `SYNTH_PLR_DEMO.db` |
| `plr_generator.py` | Core parametric PLR curve generation from physiological models |
| `artifact_injection.py` | Inject realistic artifacts (blinks, noise, gaps) into clean signals |
| `database_builder.py` | Assemble synthetic subjects into a DuckDB database |
| `privacy_validator.py` | Validate synthetic data is not similar to real patient data |

## Usage

```bash
python -m src.synthetic.demo_dataset
```

## See Also

- Najjar et al. 2023 Br J Ophthalmol (PLR physiology reference)
- Park et al. 2011 Invest Ophthalmol Vis Sci (PIPR in glaucoma)
