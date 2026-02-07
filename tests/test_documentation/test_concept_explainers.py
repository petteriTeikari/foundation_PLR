"""
TDD tests for concept explainers documentation.

These tests verify that concept explainers exist for non-SWE audiences
(ophthalmology PIs, biostatisticians, research scientists).
"""

from pathlib import Path

import pytest


@pytest.fixture
def concepts_doc_path():
    """Path to concept explainers document."""
    project_root = Path(__file__).parent.parent.parent
    return project_root / "docs" / "concepts-for-researchers.md"


@pytest.fixture
def concepts_content(concepts_doc_path):
    """Content of concepts document."""
    if not concepts_doc_path.exists():
        pytest.skip("Concept explainers not yet created")
    return concepts_doc_path.read_text()


class TestConceptsDocExists:
    """Test that concept explainers document exists."""

    def test_concepts_doc_exists(self, concepts_doc_path):
        """Concept explainers should exist."""
        assert concepts_doc_path.exists(), "concepts-for-researchers.md should exist"


class TestRequiredConcepts:
    """Test that all required concepts are explained."""

    @pytest.mark.parametrize(
        "concept",
        [
            "MLflow",
            "Hydra",
            "DuckDB",
            "pytest",
            "pre-commit",
        ],
    )
    def test_concept_explained(self, concepts_content, concept):
        """Each concept should be explained."""
        assert concept.lower() in concepts_content.lower(), (
            f"{concept} should be explained"
        )

    def test_has_eli5_explanations(self, concepts_content):
        """Should have simple explanations."""
        # Check for analogy-based explanations
        simple_words = ["like", "similar", "think of", "imagine", "Excel", "Word"]
        has_analogy = any(
            word.lower() in concepts_content.lower() for word in simple_words
        )
        assert has_analogy, "Should use analogies for simple explanations"


class TestTargetAudience:
    """Test that document addresses target audiences."""

    @pytest.mark.parametrize(
        "audience_term",
        [
            "PI",
            "biostatistician",
            "researcher",
            "scientist",
        ],
    )
    def test_mentions_audience(self, concepts_content, audience_term):
        """Should mention target audiences."""
        assert (
            audience_term.lower() in concepts_content.lower()
            or "non-technical" in concepts_content.lower()
            or "clinical" in concepts_content.lower()
        ), f"Should address {audience_term} audience"


class TestPracticalExamples:
    """Test that document has practical context."""

    def test_has_why_section(self, concepts_content):
        """Should explain WHY these tools are used."""
        assert "why" in concepts_content.lower(), "Should explain why tools are needed"

    def test_has_comparison_table(self, concepts_content):
        """Should have comparison table for quick reference."""
        assert "|" in concepts_content, "Should have markdown table"
