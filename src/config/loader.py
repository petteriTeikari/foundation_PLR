"""
YAML Configuration Loader for Foundation PLR.

Provides a unified interface for loading YAML configuration files
with caching, validation, and immutable access.
"""

from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Union

import yaml
from loguru import logger


class YAMLConfigLoader:
    """
    Load and manage YAML configuration files.

    Features:
    - Lazy loading with caching
    - Immutable access via MappingProxyType
    - Path resolution relative to project root
    - Validation hooks

    Usage:
        loader = YAMLConfigLoader()
        combos = loader.load_combos()
        standard = loader.get_standard_combos()
    """

    def __init__(
        self,
        config_dir: Optional[Path] = None,
        project_root: Optional[Path] = None,
    ):
        """
        Initialize the config loader.

        Args:
            config_dir: Directory containing config files.
                        Defaults to PROJECT_ROOT/configs/VISUALIZATION
            project_root: Project root directory.
                          Defaults to parent of src/config/
        """
        if project_root is None:
            # Resolve relative to this file: src/config/loader.py -> project root
            project_root = Path(__file__).parent.parent.parent

        self.project_root = project_root

        if config_dir is None:
            config_dir = project_root / "configs" / "VISUALIZATION"

        self.config_dir = config_dir
        self._cache: Dict[str, Any] = {}

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """
        Load a YAML file with caching.

        Args:
            filename: Name of the YAML file (with or without .yaml extension)

        Returns:
            Dictionary containing the YAML contents

        Raises:
            FileNotFoundError: If the file doesn't exist
            yaml.YAMLError: If the file is invalid YAML
        """
        if not filename.endswith(".yaml"):
            filename = f"{filename}.yaml"

        filepath = self.config_dir / filename

        # Check cache
        cache_key = str(filepath)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Load file
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        logger.debug(f"Loading config from {filepath}")

        with open(filepath) as f:
            data = yaml.safe_load(f)

        # Cache and return
        self._cache[cache_key] = data
        return data

    def load_combos(self) -> MappingProxyType:
        """
        Load hyperparameter combinations.

        Returns:
            Immutable mapping of combo configurations
        """
        data = self._load_yaml("plot_hyperparam_combos")
        return MappingProxyType(data)

    def load_methods(self) -> MappingProxyType:
        """
        Load method definitions.

        Returns:
            Immutable mapping of method configurations
        """
        data = self._load_yaml("methods")
        return MappingProxyType(data)

    def load_figure_registry(self) -> MappingProxyType:
        """
        Load figure registry.

        Returns:
            Immutable mapping of figure configurations
        """
        data = self._load_yaml("figure_registry")
        return MappingProxyType(data)

    def get_standard_combos(self) -> List[Dict[str, Any]]:
        """
        Get the list of standard (main figure) combinations.

        Returns:
            List of 4 standard combo dictionaries
        """
        combos = self.load_combos()
        return list(combos.get("standard_combos", []))

    def get_extended_combos(self) -> List[Dict[str, Any]]:
        """
        Get the list of extended (supplementary) combinations.

        Returns:
            List of extended combo dictionaries
        """
        combos = self.load_combos()
        return list(combos.get("extended_combos", []))

    def get_all_combos(self) -> List[Dict[str, Any]]:
        """
        Get all combinations (standard + extended).

        Returns:
            Combined list of all combo dictionaries
        """
        return self.get_standard_combos() + self.get_extended_combos()

    def get_combo_by_id(self, combo_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific combo by its ID.

        Args:
            combo_id: The combo identifier (e.g., "ground_truth", "best_ensemble")

        Returns:
            The combo dictionary or None if not found
        """
        for combo in self.get_all_combos():
            if combo.get("id") == combo_id:
                return combo
        return None

    def load_colors(self) -> MappingProxyType:
        """
        Load color definitions.

        Returns:
            Immutable mapping of color configurations
        """
        data = self._load_yaml("colors")
        return MappingProxyType(data)

    def get_method_color(
        self, method_name: str, method_type: str = "outlier_detection"
    ) -> Optional[str]:
        """
        Get the color for a specific method.

        Args:
            method_name: Name of the method (e.g., "pupil-gt", "LOF")
            method_type: Type of method ("outlier_detection" or "imputation")

        Returns:
            Hex color code or None if not found
        """
        try:
            colors = self.load_colors()

            # Map method names to color keys
            # Ground truth methods
            if method_name in ("pupil-gt", "gt", "ground_truth"):
                return colors.get("ground_truth") or colors.get("combo_ground_truth")

            # LOF and traditional methods
            if method_name in ("LOF", "OneClassSVM", "PROPHET", "SubPCA"):
                return colors.get("traditional")

            # Foundation model methods
            if "MOMENT" in method_name or "UniTS" in method_name:
                return colors.get("fm_primary")

            if "TimesNet" in method_name:
                return colors.get("timesnet")

            # Ensemble methods
            if "ensemble" in method_name.lower():
                return colors.get("ensemble")

            # Deep learning imputation
            if method_name in ("SAITS", "CSDI"):
                return colors.get("dl_primary")

            # Fallback to category colors
            return colors.get("category_default")

        except FileNotFoundError:
            logger.warning(f"Colors config not found, returning None for {method_name}")
            return None

    def clear_cache(self) -> None:
        """Clear the configuration cache."""
        self._cache.clear()
        logger.debug("Config cache cleared")


def load_yaml_file(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Convenience function to load a single YAML file.

    Args:
        filepath: Path to the YAML file

    Returns:
        Dictionary containing the YAML contents
    """
    filepath = Path(filepath)
    with open(filepath) as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path to the project root
    """
    return Path(__file__).parent.parent.parent
