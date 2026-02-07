#!/usr/bin/env python3
"""
Visual Figure Validation Script.

Checks generated figures for common quality issues:
1. File existence and non-zero size
2. Edge clipping detection (content at image borders)
3. Minimum dimensions check
4. Background color consistency

Usage:
    python scripts/validate_figures.py [--fix] [--verbose]

Author: Foundation PLR Team
Date: 2026-01-25
"""

import argparse
import sys
from pathlib import Path

# Try to import PIL
try:
    import numpy as np
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("WARNING: Pillow not installed. Run: uv pip install Pillow")
    print("         Only basic file checks will be performed.\n")


# Configuration
FIGURE_DIR = Path("figures/generated/ggplot2")
EXPECTED_BG_COLOR = (251, 249, 243)  # #FBF9F3 off-white
BG_TOLERANCE = 10  # Allow slight color variation
MIN_WIDTH = 400
MIN_HEIGHT = 300
EDGE_MARGIN = 5  # Pixels from edge to check for clipping

# Graduated edge thresholds (based on reviewer recommendations)
# Different edges have different tolerance for non-background content
EDGE_THRESHOLDS = {
    "right": 0.03,  # 3% - Legends clip here, be strict
    "bottom": 0.05,  # 5% - Captions/annotations
    "top": 0.05,  # 5% - Titles
    "left": 0.08,  # 8% - Y-axis labels often touch edge intentionally
}


class FigureValidator:
    """Validates generated figures for quality issues."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.issues = []
        self.warnings = []

    def log(self, msg: str) -> None:
        """Log verbose messages."""
        if self.verbose:
            print(f"  {msg}")

    def check_file_exists(self, path: Path) -> bool:
        """Check if file exists and has content."""
        if not path.exists():
            self.issues.append(f"MISSING: {path.name}")
            return False

        size = path.stat().st_size
        if size == 0:
            self.issues.append(f"EMPTY: {path.name} (0 bytes)")
            return False

        self.log(f"File OK: {path.name} ({size / 1024:.1f} KB)")
        return True

    def check_dimensions(self, img: "Image.Image", path: Path) -> bool:
        """Check minimum dimensions."""
        width, height = img.size
        if width < MIN_WIDTH or height < MIN_HEIGHT:
            self.warnings.append(
                f"SMALL: {path.name} ({width}x{height}, min {MIN_WIDTH}x{MIN_HEIGHT})"
            )
            return False

        self.log(f"Dimensions OK: {width}x{height}")
        return True

    def check_edge_clipping(self, img: "Image.Image", path: Path) -> bool:
        """
        Check for content clipping at edges.

        If non-background pixels are found at the image edges,
        it suggests content may be cut off.

        Uses graduated thresholds per edge (reviewer recommendation):
        - Right: 3% (legends clip here)
        - Bottom/Top: 5% (captions, titles)
        - Left: 8% (y-axis labels often touch)
        """
        arr = np.array(img.convert("RGB"))
        height, width = arr.shape[:2]

        def is_background(pixel: np.ndarray) -> bool:
            """Check if pixel is close to expected background color."""
            return all(
                abs(int(pixel[i]) - EXPECTED_BG_COLOR[i]) <= BG_TOLERANCE
                for i in range(3)
            )

        def check_edge(edge_pixels: np.ndarray, edge_name: str) -> bool:
            """Check if edge has non-background content using graduated thresholds."""
            non_bg_count = sum(1 for p in edge_pixels if not is_background(p))
            total = len(edge_pixels)
            ratio = non_bg_count / total if total > 0 else 0

            threshold = EDGE_THRESHOLDS.get(edge_name, 0.05)
            if ratio > threshold:
                self.warnings.append(
                    f"CLIPPING?: {path.name} - {edge_name} edge has {ratio:.0%} content (threshold: {threshold:.0%})"
                )
                return False
            return True

        ok = True

        # Check all four edges with graduated thresholds
        edges = {
            "right": arr[:, -EDGE_MARGIN:, :].reshape(-1, 3),
            "bottom": arr[-EDGE_MARGIN:, :, :].reshape(-1, 3),
            "top": arr[:EDGE_MARGIN, :, :].reshape(-1, 3),
            "left": arr[:, :EDGE_MARGIN, :].reshape(-1, 3),
        }

        for edge_name, edge_pixels in edges.items():
            if not check_edge(edge_pixels, edge_name):
                ok = False

        if ok:
            self.log("Edge clipping check: OK (graduated thresholds)")

        return ok

    def check_background_color(self, img: "Image.Image", path: Path) -> bool:
        """Check if background color matches expected."""
        arr = np.array(img.convert("RGB"))

        # Sample corners (should be background)
        corners = [
            arr[5, 5],  # top-left
            arr[5, -5],  # top-right
            arr[-5, 5],  # bottom-left
            arr[-5, -5],  # bottom-right
        ]

        for i, corner in enumerate(corners):
            if not all(
                abs(int(corner[j]) - EXPECTED_BG_COLOR[j]) <= BG_TOLERANCE
                for j in range(3)
            ):
                corner_names = ["top-left", "top-right", "bottom-left", "bottom-right"]
                self.warnings.append(
                    f"BG COLOR: {path.name} - {corner_names[i]} corner "
                    f"is RGB{tuple(corner)}, expected ~{EXPECTED_BG_COLOR}"
                )
                return False

        self.log(f"Background color OK: ~{EXPECTED_BG_COLOR}")
        return True

    def validate_figure(self, path: Path) -> dict:
        """Run all validation checks on a figure."""
        result = {
            "path": path,
            "exists": False,
            "dimensions_ok": None,
            "no_clipping": None,
            "bg_color_ok": None,
        }

        # Basic file check
        if not self.check_file_exists(path):
            return result
        result["exists"] = True

        # Skip image analysis if PIL not available
        if not HAS_PIL:
            return result

        # Only check PNG files (PDF requires different handling)
        if path.suffix.lower() != ".png":
            self.log(f"Skipping image checks for {path.suffix} file")
            return result

        try:
            img = Image.open(path)
            result["dimensions_ok"] = self.check_dimensions(img, path)
            result["no_clipping"] = self.check_edge_clipping(img, path)
            result["bg_color_ok"] = self.check_background_color(img, path)
        except Exception as e:
            self.issues.append(f"ERROR: {path.name} - {e}")

        return result

    def validate_all(self, figure_dir: Path) -> list[dict]:
        """Validate all PNG figures in directory."""
        results = []

        png_files = sorted(figure_dir.glob("*.png"))
        if not png_files:
            print(f"No PNG files found in {figure_dir}")
            return results

        print(f"Validating {len(png_files)} figures in {figure_dir}\n")

        for png_path in png_files:
            print(f"Checking: {png_path.name}")
            result = self.validate_figure(png_path)
            results.append(result)
            print()

        return results

    def print_summary(self) -> bool:
        """Print validation summary. Returns True if all OK."""
        print("=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)

        if self.issues:
            print(f"\n{len(self.issues)} ISSUES (must fix):")
            for issue in self.issues:
                print(f"  - {issue}")

        if self.warnings:
            print(f"\n{len(self.warnings)} WARNINGS (review recommended):")
            for warning in self.warnings:
                print(f"  - {warning}")

        if not self.issues and not self.warnings:
            print("\nAll figures passed validation.")
            return True

        print()
        return len(self.issues) == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate generated figures")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--dir", type=Path, default=FIGURE_DIR, help="Figure directory")
    args = parser.parse_args()

    if not args.dir.exists():
        print(f"ERROR: Directory not found: {args.dir}")
        print("Generate figures first with: Rscript src/r/figures/fig_*.R")
        sys.exit(1)

    validator = FigureValidator(verbose=args.verbose)
    validator.validate_all(args.dir)
    success = validator.print_summary()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
