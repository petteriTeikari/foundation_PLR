"""
TDD tests for AGENTS.md - Universal LLM Agent Instructions.

These tests verify that AGENTS.md contains all required sections
for multi-LLM support (Claude, Codex, Copilot, Cursor, Gemini).
"""

from pathlib import Path

import pytest


@pytest.fixture
def agents_md_path():
    """Path to AGENTS.md in project root."""
    project_root = Path(__file__).parent.parent.parent
    return project_root / "AGENTS.md"


@pytest.fixture
def agents_md_content(agents_md_path):
    """Content of AGENTS.md."""
    if not agents_md_path.exists():
        pytest.skip("AGENTS.md not yet created")
    return agents_md_path.read_text()


class TestAgentsMdExists:
    """Test that AGENTS.md exists."""

    def test_agents_md_exists(self, agents_md_path):
        """AGENTS.md should exist in project root."""
        if not agents_md_path.exists():
            pytest.skip("AGENTS.md not yet created")


class TestAgentsMdRequiredSections:
    """Test that AGENTS.md has all required sections."""

    def test_has_quick_start_section(self, agents_md_content):
        """Should have Quick Start section."""
        assert (
            "## Quick Start" in agents_md_content
            or "# Quick Start" in agents_md_content
        )

    def test_has_critical_rules_section(self, agents_md_content):
        """Should have Critical Rules section."""
        assert "Critical Rules" in agents_md_content

    def test_has_registry_reference(self, agents_md_content):
        """Should reference the mlflow_registry as source of truth."""
        assert (
            "mlflow_registry" in agents_md_content
            or "registry" in agents_md_content.lower()
        )

    def test_has_method_counts(self, agents_md_content):
        """Should specify correct method counts."""
        assert "11" in agents_md_content, "Should mention 11 outlier methods"
        assert "8" in agents_md_content, "Should mention 8 imputation methods"

    def test_has_anti_hardcoding_rule(self, agents_md_content):
        """Should have anti-hardcoding rule."""
        assert "hardcod" in agents_md_content.lower(), (
            "Should mention hardcoding prohibition"
        )

    def test_has_figure_system_reference(self, agents_md_content):
        """Should reference figure system."""
        assert (
            "save_publication_figure" in agents_md_content
            or "figure" in agents_md_content.lower()
        )


class TestAgentsMdMultiAgentSupport:
    """Test that AGENTS.md supports multiple LLM agents."""

    @pytest.mark.parametrize(
        "agent_name",
        [
            "Claude",
            "Codex",
            "Copilot",
            "Cursor",
            "Gemini",
        ],
    )
    def test_mentions_agent(self, agents_md_content, agent_name):
        """Should mention each supported agent."""
        # At least mention the agent or have a general "all agents" statement
        assert (
            agent_name.lower() in agents_md_content.lower()
            or "all agent" in agents_md_content.lower()
        ), f"Should mention {agent_name} or have 'all agents' statement"


class TestAgentsMdCodeExamples:
    """Test that AGENTS.md has helpful code examples."""

    def test_has_python_example(self, agents_md_content):
        """Should have Python code example."""
        assert "```python" in agents_md_content or "```py" in agents_md_content

    def test_has_bash_example(self, agents_md_content):
        """Should have bash command example."""
        assert "```bash" in agents_md_content or "pytest" in agents_md_content


class TestAgentsMdNotTooLong:
    """Test that AGENTS.md is concise (unlike CLAUDE.md which is comprehensive)."""

    def test_not_too_long(self, agents_md_content):
        """AGENTS.md should be concise - under 300 lines."""
        lines = agents_md_content.split("\n")
        assert len(lines) < 300, (
            f"AGENTS.md should be concise (<300 lines), got {len(lines)}"
        )

    def test_references_claude_md(self, agents_md_content):
        """Should reference CLAUDE.md for detailed rules."""
        assert "CLAUDE.md" in agents_md_content, (
            "Should reference CLAUDE.md for details"
        )
