# src/data_io/validation/normalization_validator.py
"""
Normalization validation for PLR data.

Detects subjects with scaling anomalies (e.g., baseline not subtracted)
and provides utilities for data quality checks.

Usage:
    from src.data_io.validation import NormalizationValidator

    validator = NormalizationValidator(db_path)
    anomalies = validator.detect_all_anomalies()
    validator.print_report()
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class AnomalySeverity(Enum):
    """Severity levels for scaling anomalies."""

    WARNING = "warning"  # Offset 20-50, review recommended
    CRITICAL = "critical"  # Offset > 50, data corruption


@dataclass
class ScalingAnomaly:
    """Represents a detected scaling anomaly for a subject."""

    subject_code: str
    mean_offset: float
    severity: AnomalySeverity
    description: str
    is_known: bool = False  # True if in known anomalies list

    def __str__(self) -> str:
        status = "[KNOWN]" if self.is_known else "[NEW]"
        return f"{status} {self.subject_code}: offset={self.mean_offset:.1f} ({self.severity.value})"


# Known anomalies with documentation
# These are documented issues that haven't been fixed at source yet
KNOWN_SCALING_ANOMALIES = {
    "PLR4018": {
        "description": "pupil_orig has +184 offset - baseline not subtracted during ingestion",
        "discovered": "2026-01-31",
        "workaround": "Use pupil_raw instead of pupil_orig for visualization",
    }
}


def get_known_anomalies() -> Dict[str, Dict[str, str]]:
    """Return dictionary of known scaling anomalies."""
    return KNOWN_SCALING_ANOMALIES.copy()


class NormalizationValidator:
    """Validates normalization consistency across subjects in PLR database."""

    # Thresholds for anomaly detection
    WARNING_THRESHOLD = 20  # Offset 20-50 is warning
    CRITICAL_THRESHOLD = 50  # Offset > 50 is critical

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize validator with database path.

        Args:
            db_path: Path to DuckDB database. If None, uses default location.
        """
        self.db_path = db_path or self._find_database()
        self.anomalies: List[ScalingAnomaly] = []
        self._connection: Optional[Any] = None

    def _find_database(self) -> Path:
        """Find the PLR database in standard locations."""
        from src.utils.paths import get_seri_db_path

        # Try centralized path utility first
        try:
            db_path = get_seri_db_path()
            if db_path.exists():
                return db_path
        except Exception:
            pass

        # Fallback to relative path
        candidates = [
            Path(__file__).parents[4] / "SERI_PLR_GLAUCOMA.db",
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError("Could not find SERI_PLR_GLAUCOMA.db database")

    def _get_connection(self) -> Any:
        """Get or create database connection."""
        if self._connection is None:
            import duckdb

            self._connection = duckdb.connect(str(self.db_path), read_only=True)
        return self._connection

    def close(self) -> None:
        """Close database connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def detect_offset_anomalies(self) -> List[ScalingAnomaly]:
        """Detect subjects with abnormal pupil_orig - pupil_raw offset.

        Returns:
            List of ScalingAnomaly objects for subjects with offset > WARNING_THRESHOLD
        """
        conn = self._get_connection()

        query = """
        SELECT
            subject_code,
            AVG(pupil_orig - pupil_raw) as mean_offset,
            COUNT(*) as n_points
        FROM (
            SELECT subject_code, pupil_orig, pupil_raw FROM train
            WHERE pupil_orig IS NOT NULL AND pupil_raw IS NOT NULL
            UNION ALL
            SELECT subject_code, pupil_orig, pupil_raw FROM test
            WHERE pupil_orig IS NOT NULL AND pupil_raw IS NOT NULL
        )
        GROUP BY subject_code
        HAVING ABS(AVG(pupil_orig - pupil_raw)) > ?
        ORDER BY ABS(mean_offset) DESC
        """

        df = conn.execute(query, [self.WARNING_THRESHOLD]).fetchdf()

        anomalies = []
        for _, row in df.iterrows():
            subject = row["subject_code"]
            offset = row["mean_offset"]

            # Determine severity
            if abs(offset) > self.CRITICAL_THRESHOLD:
                severity = AnomalySeverity.CRITICAL
            else:
                severity = AnomalySeverity.WARNING

            # Check if known
            is_known = subject in KNOWN_SCALING_ANOMALIES
            description = (
                KNOWN_SCALING_ANOMALIES[subject]["description"]
                if is_known
                else f"Detected offset of {offset:.1f} between pupil_orig and pupil_raw"
            )

            anomalies.append(
                ScalingAnomaly(
                    subject_code=subject,
                    mean_offset=offset,
                    severity=severity,
                    description=description,
                    is_known=is_known,
                )
            )

        self.anomalies = anomalies
        return anomalies

    def detect_mean_anomalies(self) -> List[ScalingAnomaly]:
        """Detect subjects with abnormal pupil_orig mean values.

        Mean > 50 indicates baseline was not subtracted.
        """
        conn = self._get_connection()

        query = """
        SELECT
            subject_code,
            AVG(pupil_orig) as mean_val
        FROM (
            SELECT subject_code, pupil_orig FROM train
            UNION ALL
            SELECT subject_code, pupil_orig FROM test
        )
        GROUP BY subject_code
        HAVING AVG(pupil_orig) > 50
        """

        df = conn.execute(query).fetchdf()

        anomalies = []
        for _, row in df.iterrows():
            subject = row["subject_code"]
            mean_val = row["mean_val"]

            is_known = subject in KNOWN_SCALING_ANOMALIES

            anomalies.append(
                ScalingAnomaly(
                    subject_code=subject,
                    mean_offset=mean_val,
                    severity=AnomalySeverity.CRITICAL,
                    description=f"pupil_orig mean={mean_val:.1f} > 50 (baseline not subtracted)",
                    is_known=is_known,
                )
            )

        return anomalies

    def detect_all_anomalies(self) -> List[ScalingAnomaly]:
        """Run all anomaly detection methods and return combined results.

        Returns:
            List of all detected anomalies (deduplicated by subject)
        """
        offset_anomalies = self.detect_offset_anomalies()
        mean_anomalies = self.detect_mean_anomalies()

        # Combine and deduplicate (keep more severe)
        by_subject = {}
        for anomaly in offset_anomalies + mean_anomalies:
            existing = by_subject.get(anomaly.subject_code)
            if existing is None:
                by_subject[anomaly.subject_code] = anomaly
            elif anomaly.severity == AnomalySeverity.CRITICAL:
                by_subject[anomaly.subject_code] = anomaly

        self.anomalies = list(by_subject.values())
        return self.anomalies

    def get_new_anomalies(self) -> List[ScalingAnomaly]:
        """Return only anomalies that are NOT in the known list."""
        if not self.anomalies:
            self.detect_all_anomalies()
        return [a for a in self.anomalies if not a.is_known]

    def get_critical_anomalies(self) -> List[ScalingAnomaly]:
        """Return only CRITICAL severity anomalies."""
        if not self.anomalies:
            self.detect_all_anomalies()
        return [a for a in self.anomalies if a.severity == AnomalySeverity.CRITICAL]

    def print_report(self) -> str:
        """Print and return a formatted report of all anomalies."""
        if not self.anomalies:
            self.detect_all_anomalies()

        lines = [
            "=" * 60,
            "NORMALIZATION VALIDATION REPORT",
            "=" * 60,
            f"Database: {self.db_path}",
            f"Total anomalies detected: {len(self.anomalies)}",
            "",
        ]

        new_anomalies = self.get_new_anomalies()
        known_anomalies = [a for a in self.anomalies if a.is_known]

        if new_anomalies:
            lines.append("NEW ANOMALIES (require investigation):")
            lines.append("-" * 40)
            for a in new_anomalies:
                lines.append(f"  {a}")
            lines.append("")

        if known_anomalies:
            lines.append("KNOWN ANOMALIES (documented):")
            lines.append("-" * 40)
            for a in known_anomalies:
                lines.append(f"  {a}")
                lines.append(f"    {a.description}")
            lines.append("")

        if not self.anomalies:
            lines.append("No anomalies detected.")

        lines.append("=" * 60)

        report = "\n".join(lines)
        print(report)
        return report

    def validate_or_raise(self) -> None:
        """Validate data and raise exception if NEW critical anomalies found.

        Use this in data loading pipelines to enforce data quality.

        Raises:
            ValueError: If new critical anomalies are detected
        """
        self.detect_all_anomalies()
        new_critical = [
            a
            for a in self.anomalies
            if not a.is_known and a.severity == AnomalySeverity.CRITICAL
        ]

        if new_critical:
            msg = "New critical scaling anomalies detected:\n"
            msg += "\n".join(f"  - {a}" for a in new_critical)
            msg += "\n\nFix at source or add to KNOWN_SCALING_ANOMALIES."
            raise ValueError(msg)


def validate_subject_scaling(
    subject_code: str,
    pupil_orig: List[float],
    pupil_raw: List[float],
) -> Optional[ScalingAnomaly]:
    """Validate scaling for a single subject's data.

    Args:
        subject_code: Subject identifier
        pupil_orig: Original pupil signal values
        pupil_raw: Raw (NaN-masked) pupil signal values

    Returns:
        ScalingAnomaly if anomaly detected, None otherwise
    """
    import numpy as np

    # Filter out NaN/None
    valid_pairs = [
        (o, r)
        for o, r in zip(pupil_orig, pupil_raw)
        if o is not None and r is not None and not np.isnan(o) and not np.isnan(r)
    ]

    if not valid_pairs:
        return None

    orig_vals, raw_vals = zip(*valid_pairs)
    mean_offset = np.mean(np.array(orig_vals) - np.array(raw_vals))

    if abs(mean_offset) > NormalizationValidator.CRITICAL_THRESHOLD:
        severity = AnomalySeverity.CRITICAL
    elif abs(mean_offset) > NormalizationValidator.WARNING_THRESHOLD:
        severity = AnomalySeverity.WARNING
    else:
        return None  # No anomaly

    is_known = subject_code in KNOWN_SCALING_ANOMALIES

    return ScalingAnomaly(
        subject_code=subject_code,
        mean_offset=float(mean_offset),
        severity=severity,
        description=f"Offset of {mean_offset:.1f} detected",
        is_known=is_known,
    )


if __name__ == "__main__":
    # Run validation report
    validator = NormalizationValidator()
    validator.print_report()
    validator.close()
