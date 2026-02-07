# Optimized Figure Assets

This directory contains **web-optimized JPEG versions** of the repository documentation figures.

## Why Optimized Images?

The original PNGs in `../generated/` are ~7MB each (suitable for publication). For web documentation:

| Original PNG | Optimized JPEG |
|--------------|----------------|
| ~7MB | ~150-200KB |
| 2816x1536px | 1600px width |
| Sharp corners | 24px rounded corners |
| Slow loading | Fast loading |

## Regenerating Images

To regenerate optimized images from source PNGs:

```bash
# Process all images
.venv/bin/python docs/repo-figures/scripts/resize_and_convert.py

# Process single image
.venv/bin/python docs/repo-figures/scripts/resize_and_convert.py --input fig-repo-01-what-this-repo-does.png

# Preview without saving
.venv/bin/python docs/repo-figures/scripts/resize_and_convert.py --dry-run
```

## Placeholder Images

Small (16x16 black) placeholder images exist for figures not yet generated from Nano Banana Pro:

- `fig-repo-17-logging-levels.jpg`
- `fig-repo-33-decomposition-grid.jpg`
- `fig-repo-41-dca-expert-mechanics.jpg`
- `fig-repo-42-dca-threshold-sensitivity.jpg`
- `fig-repro-08c-dim-reduction-example.jpg`
- `fig-repro-12-dependency-explosion.jpg`
- `fig-trans-15-plr-code-domain-specific.jpg`
- `fig-trans-16-configuration-vs-hardcoding.jpg`
- `fig-trans-17-registry-pattern.jpg`
- `fig-trans-18-fork-guide.jpg`
- `fig-trans-19-data-quality-manifesto.jpg`
- `fig-trans-20-choose-your-approach.jpg`

These will be replaced when actual figures are generated.

## Alt Text & Captions

SEO/GEO-optimized alt text and compact captions for figures 61-98 are documented in `../figure-alt-text-catalog.md`. Use those when embedding figures in README files. See GH#16 for the audit of figures 01-60.

## Using Images in Documentation

Reference images relative to the documentation file:

```markdown
![What This Repo Does: Overview of the foundation_PLR preprocessing pipeline](../../docs/repo-figures/assets/fig-repo-01-what-this-repo-does.jpg)
```

See `docs/planning/repo-visual-documentation-plan.md` for the full Hebbian placement strategy.
