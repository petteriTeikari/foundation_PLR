"""
Guardrail Tests: JSON Provenance Metadata

All JSON exports must have complete provenance metadata for reproducibility.
"""

import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.data, pytest.mark.guardrail]

PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestJSONProvenance:
    """Test that all JSON exports have proper provenance metadata."""

    def get_json_files(self):
        """Get all JSON data files."""
        r_data_dir = PROJECT_ROOT / "data" / "r_data"
        if not r_data_dir.exists():
            return []
        return list(r_data_dir.glob("*.json"))

    def test_all_json_have_metadata(self):
        """All JSON files must have a metadata section."""
        json_files = self.get_json_files()
        if not json_files:
            pytest.skip("No JSON files found in data/r_data/")

        violations = []
        for json_file in json_files:
            try:
                data = json.loads(json_file.read_text())
                if "metadata" not in data:
                    violations.append(
                        {
                            "file": json_file.name,
                            "issue": 'Missing "metadata" key',
                        }
                    )
            except json.JSONDecodeError as e:
                violations.append(
                    {
                        "file": json_file.name,
                        "issue": f"Invalid JSON: {e}",
                    }
                )

        if violations:
            msg = "GUARDRAIL VIOLATION: JSON files missing metadata!\n\n"
            for v in violations:
                msg += f"  {v['file']}: {v['issue']}\n"
            pytest.fail(msg)

    def test_json_have_data_source(self):
        """All JSON files must have data_source in metadata."""
        json_files = self.get_json_files()
        if not json_files:
            pytest.skip("No JSON files found")

        violations = []
        for json_file in json_files:
            try:
                data = json.loads(json_file.read_text())
                metadata = data.get("metadata", {})
                if "data_source" not in metadata:
                    violations.append(
                        {
                            "file": json_file.name,
                            "issue": 'Missing "data_source" in metadata',
                        }
                    )
            except json.JSONDecodeError:
                pass  # Already caught in previous test

        if violations:
            msg = "GUARDRAIL VIOLATION: JSON files missing data_source!\n\n"
            for v in violations:
                msg += f"  {v['file']}: {v['issue']}\n"
            msg += "\nFIX: Add data_source with database path and hash."
            pytest.fail(msg)

    def test_json_have_db_hash(self):
        """All JSON files must have database hash for reproducibility."""
        json_files = self.get_json_files()
        if not json_files:
            pytest.skip("No JSON files found")

        violations = []
        for json_file in json_files:
            try:
                data = json.loads(json_file.read_text())
                data_source = data.get("metadata", {}).get("data_source", {})
                if isinstance(data_source, dict) and "db_hash" not in data_source:
                    # Allow if data_source is a string (simple reference)
                    violations.append(
                        {
                            "file": json_file.name,
                            "issue": 'Missing "db_hash" in data_source',
                        }
                    )
            except json.JSONDecodeError:
                pass

        if violations:
            msg = "GUARDRAIL VIOLATION: JSON files missing db_hash!\n\n"
            for v in violations:
                msg += f"  {v['file']}: {v['issue']}\n"
            msg += "\nFIX: Add hash of source database file for audit trail."
            pytest.fail(msg)

    def test_json_have_generator_info(self):
        """All JSON files should have generator script info."""
        json_files = self.get_json_files()
        if not json_files:
            pytest.skip("No JSON files found")

        violations = []
        for json_file in json_files:
            try:
                data = json.loads(json_file.read_text())
                metadata = data.get("metadata", {})
                if "generator" not in metadata:
                    violations.append(
                        {
                            "file": json_file.name,
                            "issue": 'Missing "generator" in metadata',
                        }
                    )
            except json.JSONDecodeError:
                pass

        if violations:
            msg = "GUARDRAIL VIOLATION: JSON files missing generator info!\n\n"
            for v in violations:
                msg += f"  {v['file']}: {v['issue']}\n"
            msg += "\nFIX: Add generator script path to metadata."
            pytest.fail(msg)

    def test_no_synthetic_data_markers(self):
        """Check that JSON files don't indicate synthetic/fake data."""
        json_files = self.get_json_files()
        if not json_files:
            pytest.skip("No JSON files found")

        violations = []
        suspicious_terms = ["synthetic", "fake", "random", "simulated", "mock"]

        for json_file in json_files:
            content = json_file.read_text().lower()
            for term in suspicious_terms:
                if term in content:
                    violations.append(
                        {
                            "file": json_file.name,
                            "issue": f'Contains suspicious term: "{term}"',
                        }
                    )

        if violations:
            msg = "GUARDRAIL VIOLATION: Possible synthetic data in JSON files!\n\n"
            for v in violations:
                msg += f"  {v['file']}: {v['issue']}\n"
            msg += (
                "\nCRITICAL: All figure data must come from REAL experimental results."
            )
            pytest.fail(msg)
