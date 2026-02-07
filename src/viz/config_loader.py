"""Configuration loader for visualization modules.

Loads configuration from configs/VISUALIZATION/ and configs/defaults.yaml.
Uses proper YAML parsing (NOT grep/sed) per project guidelines.

This is a minimal loader for non-Hydra code paths. For main experiments,
use Hydra with @hydra.main(config_path="configs", config_name="defaults").
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""

    pass


def _find_configs_root() -> Path:
    """Find the configs/ directory."""
    # From src/viz/config_loader.py, go up to project root
    current = Path(__file__).resolve()
    for _ in range(5):  # Max 5 levels up
        configs_dir = current / "configs"
        if configs_dir.is_dir() and (configs_dir / "defaults.yaml").exists():
            return configs_dir
        current = current.parent
    raise ConfigurationError(
        f"Could not find configs/ directory with defaults.yaml. "
        f"Searched from {Path(__file__).resolve()}"
    )


@lru_cache(maxsize=1)
def _get_configs_root() -> Path:
    """Cached access to configs root."""
    return _find_configs_root()


@lru_cache(maxsize=32)
def _load_yaml_file(filepath: Path) -> Dict[str, Any]:
    """Load a YAML file with caching."""
    if not filepath.exists():
        raise ConfigurationError(f"Config file not found: {filepath}")
    with open(filepath, "r") as f:
        return yaml.safe_load(f) or {}


class ConfigLoader:
    """Centralized configuration loader for visualization code.

    Loads from:
    - configs/defaults.yaml (main config with prevalence, bootstrap params, etc.)
    - configs/VISUALIZATION/*.yaml (visualization-specific configs)
    """

    def __init__(self) -> None:
        self._root: Path = _get_configs_root()
        self._viz_root: Path = self._root / "VISUALIZATION"

    def get_defaults(self) -> Dict[str, Any]:
        """Get main defaults.yaml config."""
        return _load_yaml_file(self._root / "defaults.yaml")

    def get_viz_config(self, name: str) -> Dict[str, Any]:
        """Get a visualization config file by name (without .yaml extension)."""
        filepath = self._viz_root / f"{name}.yaml"
        return _load_yaml_file(filepath)

    # === Convenience methods for common config access ===

    def get_prevalence(self) -> float:
        """Get glaucoma disease prevalence from config."""
        defaults = self.get_defaults()
        try:
            return defaults["CLS_EVALUATION"]["glaucoma_params"]["prevalence"]
        except KeyError:
            raise ConfigurationError(
                "Missing CLS_EVALUATION.glaucoma_params.prevalence in defaults.yaml"
            )

    def get_bootstrap_params(self) -> Dict[str, Any]:
        """Get bootstrap evaluation parameters from config."""
        defaults = self.get_defaults()
        try:
            return defaults["CLS_EVALUATION"]["BOOTSTRAP"]
        except KeyError:
            raise ConfigurationError(
                "Missing CLS_EVALUATION.BOOTSTRAP in defaults.yaml"
            )

    def get_combos(self) -> Dict[str, Any]:
        """Get hyperparam combos from VISUALIZATION/plot_hyperparam_combos.yaml."""
        return self.get_viz_config("plot_hyperparam_combos")

    def get_combo_config(self, combo_name: str) -> Dict[str, Any]:
        """Get config for a specific combo by name (id)."""
        combos = self.get_combos()
        for combo_type in ["standard_combos", "extended_combos"]:
            if combo_type in combos:
                for combo in combos[combo_type]:
                    if combo.get("id") == combo_name:
                        return combo
        raise ConfigurationError(f"Combo '{combo_name}' not found in config")

    def get_standard_combo_names(self) -> List[str]:
        """Get list of standard combo names."""
        combos = self.get_combos()
        return [c.get("id") for c in combos.get("standard_combos", [])]

    def get_standard_hyperparam_combos(self) -> List[Dict[str, Any]]:
        """Get list of standard hyperparam combo configs."""
        combos = self.get_combos()
        return combos.get("standard_combos", [])

    def get_extended_hyperparam_combos(self) -> List[Dict[str, Any]]:
        """Get list of extended hyperparam combo configs."""
        combos = self.get_combos()
        return combos.get("extended_combos", [])

    def get_metrics_config(self) -> Dict[str, Any]:
        """Get metrics configuration from VISUALIZATION/metrics.yaml."""
        return self.get_viz_config("metrics")

    def get_metric_combo(self, combo_name: Optional[str] = None) -> List[str]:
        """Get list of metrics for a named combo."""
        metrics = self.get_metrics_config()
        # Use config default if not specified
        if combo_name is None:
            combo_name = metrics.get("defaults", {}).get("metric_combo", "default")
        combos = metrics.get("metric_combos", {})
        if combo_name in combos:
            return combos[combo_name].get("metrics", [])
        raise ConfigurationError(f"Metric combo '{combo_name}' not found")

    def get_metric_label(self, metric_name: str) -> str:
        """Get display label for a metric."""
        metrics = self.get_metrics_config()
        defs = metrics.get("metric_definitions", {})
        if metric_name in defs:
            return defs[metric_name].get("display_name", metric_name)
        return metric_name

    def get_colors(self) -> Dict[str, Any]:
        """Get color configuration from VISUALIZATION/colors.yaml."""
        return self.get_viz_config("colors")

    def get_combo_color(self, combo_name: str) -> str:
        """Get color for a specific combo."""
        colors = self.get_colors()
        combo_colors = colors.get("combo_colors", {})
        return combo_colors.get(combo_name, "#4A4A4A")  # Neutral gray fallback

    def get_methods_config(self) -> Dict[str, Any]:
        """Get method names and display names from VISUALIZATION/methods.yaml."""
        return self.get_viz_config("methods")

    def get_method_display_name(self, method_name: str) -> str:
        """Get display name for an outlier/imputation method."""
        methods = self.get_methods_config()

        # Check outlier methods
        for method in methods.get("outlier_detection", {}).get("methods", []):
            if method.get("name") == method_name:
                return method.get("display_name", method_name)

        # Check imputation methods
        for method in methods.get("imputation", {}).get("methods", []):
            if method.get("name") == method_name:
                return method.get("display_name", method_name)

        return method_name

    def get_display_names_config(self) -> Dict[str, Any]:
        """Get display names from configs/mlflow_registry/display_names.yaml."""
        filepath = (
            self._root.parent / "configs" / "mlflow_registry" / "display_names.yaml"
        )
        return _load_yaml_file(filepath)

    def get_category_display_names(self) -> Dict[str, str]:
        """Get category display names mapping (id -> display_name)."""
        config = self.get_display_names_config()
        categories = config.get("categories", {})
        return {
            cat_id: cat_info.get("display_name", cat_id)
            for cat_id, cat_info in categories.items()
        }

    def get_category_short_names(self) -> Dict[str, str]:
        """Get category short names mapping (display_name -> short_name)."""
        config = self.get_display_names_config()
        categories = config.get("categories", {})
        return {
            cat_info.get("display_name", cat_id): cat_info.get("short_name", cat_id)
            for cat_id, cat_info in categories.items()
        }


# Singleton instance with caching
_loader_instance: Optional[ConfigLoader] = None


def get_config_loader() -> ConfigLoader:
    """Get the singleton ConfigLoader instance."""
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = ConfigLoader()
    return _loader_instance


def clear_config_cache() -> None:
    """Clear all cached configuration data.

    Useful for testing or when config files are modified.
    """
    global _loader_instance
    _loader_instance = None
    _load_yaml_file.cache_clear()
    _get_configs_root.cache_clear()
