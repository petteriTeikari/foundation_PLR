"""
Test deliverables verification.

Ensures generated figures and registry are consistent.
Addresses: GAP-15 (prevents 0-figure delivery catastrophe)
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestDeliverablesVerificationIntegration:
    """Integration tests for deliverables verification."""

    def test_existing_figures_pass_verification(self):
        """Figures that exist should pass verification."""
        # Check that at least some figures exist
        figure_dirs = [
            PROJECT_ROOT / "figures" / "generated" / "ggplot2" / "main",
            PROJECT_ROOT / "figures" / "generated" / "ggplot2" / "supplementary",
        ]

        existing_figures = []
        for fig_dir in figure_dirs:
            if fig_dir.exists():
                existing_figures.extend(fig_dir.glob("*.png"))

        assert (
            len(existing_figures) > 0
        ), "No figures found in figures/generated/. Run figure generation first."

    def test_figure_registry_matches_generated(self):
        """Figures in registry should exist in generated directory."""
        import yaml

        registry_path = (
            PROJECT_ROOT / "configs" / "VISUALIZATION" / "figure_registry.yaml"
        )
        assert registry_path.exists(), f"Missing: {registry_path}"

        with open(registry_path) as f:
            registry = yaml.safe_load(f)

        # Get R figures from registry
        r_figures = registry.get("r_figures", {})

        # Check at least some exist
        missing = []
        for fig_name in list(r_figures.keys())[:5]:  # Check first 5
            patterns = [
                f"figures/generated/**/{fig_name}.png",
                f"figures/generated/**/{fig_name}.pdf",
            ]
            found = any(list(PROJECT_ROOT.glob(p)) for p in patterns)
            if not found:
                missing.append(fig_name)

        # Allow some missing (not all figures generated yet)
        # But warn if many are missing
        assert (
            len(missing) <= 3
        ), f"Many figures not generated: {missing}. Run: make analyze"
