# src/preprocess/normalization_manager.py
"""
Unified Normalization Manager for Foundation PLR.

Provides a single, stateful normalization system that:
1. Maintains consistent scaler parameters across the pipeline
2. Tracks normalization state to prevent double-transforms
3. Handles NaN values properly
4. Serializes/deserializes for persistence

Usage:
    from src.preprocess.normalization_manager import NormalizationManager

    manager = NormalizationManager()
    manager.fit(train_data)

    # Standard transforms (no state tracking)
    normalized = manager.transform(data)
    restored = manager.inverse_transform(normalized)

    # Tracked transforms (prevents double normalization)
    normalized, data_id = manager.transform_tracked(data)
    restored, _ = manager.inverse_transform_tracked(normalized, data_id)

    # Serialization
    params = manager.to_dict()
    manager2 = NormalizationManager.from_dict(params)
"""

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np


class NormalizationState(Enum):
    """Possible normalization states for tracked data."""

    RAW = "raw"
    NORMALIZED = "normalized"
    DENORMALIZED = "denormalized"


class DoubleNormalizationError(Exception):
    """Raised when attempting to normalize already-normalized data."""

    pass


class DoubleDenormalizationError(Exception):
    """Raised when attempting to denormalize already-raw data."""

    pass


class NotFittedError(Exception):
    """Raised when transform is called before fit."""

    pass


@dataclass
class NormalizationManager:
    """Unified normalization manager with state tracking.

    Implements z-score standardization: (x - mean) / std

    Attributes:
        mean: Mean value from fit data (computed with nanmean)
        std: Standard deviation from fit data (computed with nanstd, ddof=0)
        is_fitted: Whether fit() has been called
    """

    mean: Optional[float] = None
    std: Optional[float] = None
    is_fitted: bool = False
    _state_registry: dict = field(default_factory=dict)

    def fit(self, data: np.ndarray) -> "NormalizationManager":
        """Fit normalization parameters from data.

        Args:
            data: Array to compute mean and std from. NaN values are ignored.

        Returns:
            self for method chaining
        """
        self.mean = float(np.nanmean(data))
        self.std = float(np.nanstd(data, ddof=0))

        # Prevent division by zero
        if self.std == 0 or np.isnan(self.std):
            self.std = 1.0

        self.is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply z-score normalization.

        Args:
            data: Array to normalize

        Returns:
            Normalized array: (data - mean) / std
            NaN values are preserved.
        """
        if not self.is_fitted:
            raise NotFittedError("Call fit() before transform()")

        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reverse z-score normalization.

        Args:
            data: Normalized array

        Returns:
            Original-scale array: data * std + mean
            NaN values are preserved.
        """
        if not self.is_fitted:
            raise NotFittedError("Call fit() before inverse_transform()")

        return data * self.std + self.mean

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            data: Array to fit on and transform

        Returns:
            Normalized array
        """
        self.fit(data)
        return self.transform(data)

    # =========================================================================
    # STATE-TRACKED TRANSFORMS (prevent double normalization)
    # =========================================================================

    def transform_tracked(
        self,
        data: np.ndarray,
        data_id: Optional[str] = None,
    ) -> Tuple[np.ndarray, str]:
        """Transform with state tracking to prevent double normalization.

        Args:
            data: Array to normalize
            data_id: Optional identifier for this data. If None, generates new ID.

        Returns:
            Tuple of (normalized_data, data_id)

        Raises:
            DoubleNormalizationError: If data_id is already normalized
        """
        if data_id is None:
            data_id = str(uuid.uuid4())[:8]

        # Check state
        current_state = self._state_registry.get(data_id, NormalizationState.RAW)
        if current_state == NormalizationState.NORMALIZED:
            raise DoubleNormalizationError(
                f"Data '{data_id}' is already normalized. "
                "Use inverse_transform_tracked() first."
            )

        # Transform and update state
        result = self.transform(data)
        self._state_registry[data_id] = NormalizationState.NORMALIZED

        return result, data_id

    def inverse_transform_tracked(
        self,
        data: np.ndarray,
        data_id: str,
    ) -> Tuple[np.ndarray, str]:
        """Inverse transform with state tracking to prevent double denormalization.

        Args:
            data: Normalized array
            data_id: Identifier from transform_tracked()

        Returns:
            Tuple of (denormalized_data, data_id)

        Raises:
            DoubleDenormalizationError: If data_id is already raw/denormalized
        """
        current_state = self._state_registry.get(data_id)

        if current_state is None:
            # Unknown data - assume it's normalized (external source)
            pass
        elif current_state != NormalizationState.NORMALIZED:
            raise DoubleDenormalizationError(
                f"Data '{data_id}' is not normalized (state={current_state}). "
                "Cannot inverse_transform."
            )

        # Transform and update state
        result = self.inverse_transform(data)
        self._state_registry[data_id] = NormalizationState.RAW

        return result, data_id

    def is_normalized(self, data_id: str) -> bool:
        """Check if data with given ID is currently normalized.

        Args:
            data_id: Data identifier or array (for identity check)

        Returns:
            True if data is in normalized state
        """
        state = self._state_registry.get(data_id)
        return state == NormalizationState.NORMALIZED

    def clear_state(self, data_id: Optional[str] = None):
        """Clear state tracking for data.

        Args:
            data_id: Specific ID to clear, or None to clear all
        """
        if data_id is None:
            self._state_registry.clear()
        elif data_id in self._state_registry:
            del self._state_registry[data_id]

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> dict:
        """Serialize normalization parameters to dictionary.

        Returns:
            Dict with mean, std, and is_fitted
        """
        return {
            "mean": self.mean,
            "std": self.std,
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_dict(cls, params: dict) -> "NormalizationManager":
        """Create NormalizationManager from serialized dictionary.

        Args:
            params: Dict with mean, std, is_fitted

        Returns:
            New NormalizationManager instance
        """
        manager = cls()
        manager.mean = params.get("mean")
        manager.std = params.get("std")
        manager.is_fitted = params.get("is_fitted", False)
        return manager

    def __repr__(self) -> str:
        if self.is_fitted:
            return f"NormalizationManager(mean={self.mean:.4f}, std={self.std:.4f})"
        return "NormalizationManager(not fitted)"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_manager_from_data(data: np.ndarray) -> NormalizationManager:
    """Create and fit a NormalizationManager from data.

    Args:
        data: Training data to compute statistics from

    Returns:
        Fitted NormalizationManager
    """
    return NormalizationManager().fit(data)


def standardize_array(
    data: np.ndarray,
    mean: Optional[float] = None,
    std: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
    """Standardize array to z-scores.

    Args:
        data: Array to standardize
        mean: Pre-computed mean (if None, computed from data)
        std: Pre-computed std (if None, computed from data)

    Returns:
        Tuple of (standardized_data, mean, std)
    """
    if mean is None:
        mean = float(np.nanmean(data))
    if std is None:
        std = float(np.nanstd(data, ddof=0))
        if std == 0:
            std = 1.0

    standardized = (data - mean) / std
    return standardized, mean, std


def destandardize_array(
    data: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    """Reverse standardization.

    Args:
        data: Standardized array
        mean: Mean used for standardization
        std: Std used for standardization

    Returns:
        Original-scale array
    """
    return data * std + mean
