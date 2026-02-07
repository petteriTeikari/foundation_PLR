# GUARDRAILS - Quick Reference (5 Critical Rules)

## 1. HYPERPARAM COMBOS: Load from configs/VISUALIZATION/plot_hyperparam_combos.yaml
```python
combos = yaml.safe_load(open("configs/VISUALIZATION/plot_hyperparam_combos.yaml"))["standard_combos"]
```

## 2. SUBJECTS: Load from configs/demo_subjects.yaml
```python
subjects = yaml.safe_load(open("configs/demo_subjects.yaml"))["demo_subjects"]
```

## 3. GROUND TRUTH: MUST appear in EVERY comparison figure

## 4. MAX CURVES: 4 curves per main figure, 6 for supplementary

## 5. COLORS: Use semantic CSS variables, NEVER hardcode hex values
- `var(--color-ground-truth)` - gray
- `var(--color-fm-primary)` - blue
- `var(--color-traditional)` - orange
- `var(--color-baseline)` - pink
