# PLR_FEATURIZATION - Handcrafted Feature Extraction

## Purpose

Defines time-bin features extracted from preprocessed PLR signals for classification. These physiologically-motivated features capture key aspects of the pupillary light reflex response.

## Files

| File | Description |
|------|-------------|
| `featuresSimple.yaml` | Minimal feature set (4 features) |
| `featuresBaseline.yaml` | Extended feature set (6 features) |

## Feature Definition Structure

Each feature is defined by:

```yaml
FEATURE_NAME:
  time_from: 'onset' | 'offset'  # Reference point
  time_start: <seconds>           # Start of time bin
  time_end: <seconds>             # End of time bin
  measure: 'amplitude' | 'timing' # What to measure
  stat: 'min' | 'max' | 'mean' | 'median' | 'AUC'  # Aggregation
```

## Core PLR Features

### MAX_CONSTRICTION

Maximum pupil constriction after light stimulus.

```yaml
MAX_CONSTRICTION:
  time_from: 'onset'
  time_start: 0
  time_end: 15
  measure: 'amplitude'
  stat: 'min'
```

**Physiology**: Reflects combined rod, cone, and melanopsin-driven constriction. Early component dominated by rods/cones, later by ipRGCs.

### PHASIC

Initial rapid constriction phase.

```yaml
PHASIC:
  time_from: 'onset'
  time_start: 0
  time_end: 5
  measure: 'amplitude'
  stat: 'min'
```

**Physiology**: Fast component driven primarily by rod and cone pathways. Peaks ~1s post-stimulus.

### SUSTAINED

Sustained constriction near stimulus offset.

```yaml
SUSTAINED:
  time_from: 'offset'
  time_start: -5
  time_end: 0
  measure: 'amplitude'
  stat: 'min'
```

**Physiology**: Maintained constriction reflecting sustained photoreceptor input plus melanopsin contribution.

### PIPR (Post-Illumination Pupil Response)

Pupil redilation after stimulus offset.

```yaml
PIPR:
  time_from: 'offset'
  time_start: 0
  time_end: 15
  measure: 'amplitude'
  stat: 'min'
```

**Physiology**: Slow redilation driven by melanopsin ipRGC activity. Marker of ipRGC function.

### PIPR_AUC

Area under the PIPR curve.

```yaml
PIPR_AUC:
  time_from: 'offset'
  time_start: 0
  time_end: 12
  measure: 'amplitude'
  stat: 'AUC'
```

**Physiology**: Cumulative melanopsin response. May be more robust than single-point measurements.

### BASELINE

Pre-stimulus pupil size.

```yaml
BASELINE:
  time_from: 'onset'
  time_start: -5
  time_end: 0
  measure: 'amplitude'
  stat: 'median'
```

**Physiology**: Resting pupil diameter. Affected by age, ambient light, arousal state.

## Time Reference Points

| Reference | Description |
|-----------|-------------|
| `onset` | Light stimulus ON (t=0 of stimulus period) |
| `offset` | Light stimulus OFF (t=end of stimulus period) |

Negative `time_start` values indicate time BEFORE the reference point.

## Amplitude Bins vs Latency

The current implementation uses **amplitude-based features** (pupil size at time bins), not timing-based latency features.

**Why amplitude bins?**
- More robust to noise
- Comparable across subjects with different pupil sizes (when normalized)
- Clinically interpretable (directly maps to physiology)

**Latency features** (time to reach specific constriction levels) are commented out but available:

```yaml
# LATENCY:
#   time_from: 'onset'
#   time_start: 0
#   time_end: 5
#   measure: 'timing'
#   stat: 'min'
```

## Handcrafted vs Embeddings

This repository evaluates TWO featurization approaches:

| Approach | Config | AUROC |
|----------|--------|-------|
| **Handcrafted** | `PLR_FEATURIZATION/` (this) | ~0.83 |
| Embeddings | `PLR_EMBEDDING/` | ~0.74 |

**Finding**: Handcrafted features outperform foundation model embeddings by ~9 percentage points for this task.

**Why?** PLR features are physiologically motivated and capture domain-specific knowledge that generic embeddings miss.

## Hydra Usage

```bash
# Use simple features
python src/featurization/flow_featurization.py \
    PLR_FEATURIZATION=featuresSimple

# Use baseline features
python src/featurization/flow_featurization.py \
    PLR_FEATURIZATION=featuresBaseline

# Override specific feature
python src/featurization/flow_featurization.py \
    PLR_FEATURIZATION=featuresSimple \
    PLR_FEATURIZATION.FEATURES.PIPR_AUC.time_end=15
```

## Creating New Feature Sets

1. Copy an existing config:
   ```bash
   cp featuresSimple.yaml featuresCustom.yaml
   ```

2. Update metadata:
   ```yaml
   FEATURES_METADATA:
     name: 'custom'
     version: 1.0
   ```

3. Add/modify features under `FEATURES:`

4. Use via Hydra:
   ```bash
   PLR_FEATURIZATION=featuresCustom
   ```

## See Also

- Embeddings alternative: `../PLR_EMBEDDING/`
- Imputation (previous stage): `../MODELS/`
- Classification (next stage): `../CLS_MODELS/`
- Code: `src/featurization/`
- R-PLR bins: https://github.com/petteriTeikari/R-PLR/blob/master/config/bins.csv
- Najjar 2023 (original PLR study): DOI 10.1136/bjophthalmol-2021-319938

---

**Note**: Performance comparisons are documented in the manuscript, not this repository.
