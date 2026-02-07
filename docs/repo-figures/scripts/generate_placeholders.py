#!/usr/bin/env python3
"""
Generate placeholder images for figure plans that don't have PNGs yet.

Creates 16x16 pixel black images as placeholders that will be replaced
when the actual figures are generated.
"""

import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow is required. Install with: uv pip install pillow")
    sys.exit(1)


# Figures that need placeholders (plans exist but no PNG generated)
MISSING_FIGURES = [
    "fig-repo-17-logging-levels",
    "fig-repo-33-decomposition-grid",
    "fig-repo-41-dca-expert-mechanics",
    "fig-repo-42-dca-threshold-sensitivity",
    "fig-repro-08c-dim-reduction-example",
    "fig-repro-12-dependency-explosion",
    "fig-trans-15-plr-code-domain-specific",
    "fig-trans-16-configuration-vs-hardcoding",
    "fig-trans-17-registry-pattern",
    "fig-trans-18-fork-guide",
    "fig-trans-19-data-quality-manifesto",
    "fig-trans-20-choose-your-approach",
]


def create_placeholder(output_path: Path, size: tuple[int, int] = (16, 16)):
    """Create a black placeholder image."""
    img = Image.new("RGB", size, color=(0, 0, 0))
    img.save(output_path, "JPEG", quality=85)
    print(f"Created: {output_path.name}")


def main():
    # Output to assets directory
    assets_dir = Path(__file__).parent.parent / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating {len(MISSING_FIGURES)} placeholder images in {assets_dir}")
    print()

    for fig_name in MISSING_FIGURES:
        output_path = assets_dir / f"{fig_name}.jpg"
        create_placeholder(output_path)

    print()
    print(f"Done. {len(MISSING_FIGURES)} placeholders created.")
    print("Replace these when actual figures are generated from Nano Banana Pro.")


if __name__ == "__main__":
    main()
