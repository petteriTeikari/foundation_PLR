"""
CI Workflow Structure Guardrails
================================

TDD tests that enforce CI workflow conventions:
- ci.yml: PR-only (no push trigger), smart path filtering
- deploy-docs.yml: exists with correct Pages permissions
- config-integrity.yml: PR-only (no push trigger)

Written FIRST (TDD), then workflows updated to satisfy them.
"""

from pathlib import Path

import pytest
import yaml

WORKFLOWS_DIR = Path(__file__).parent.parent.parent / ".github" / "workflows"


def _load_workflow(name: str) -> dict:
    """Load and parse a workflow YAML file."""
    path = WORKFLOWS_DIR / name
    assert path.exists(), f"Workflow {name} does not exist at {path}"
    return yaml.safe_load(path.read_text())


# ============================================================================
# ci.yml — must be PR-only with smart path filtering
# ============================================================================


class TestCIWorkflow:
    """Validate ci.yml structure and triggers."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.wf = _load_workflow("ci.yml")

    def test_no_push_trigger(self):
        """ci.yml must NOT trigger on push (wasteful for doc-only changes)."""
        triggers = self.wf.get("on", self.wf.get(True, {}))
        assert "push" not in triggers, (
            "ci.yml should NOT have a push trigger. "
            "Use pull_request + workflow_dispatch only."
        )

    def test_has_pull_request_trigger(self):
        """ci.yml must trigger on pull_request to main."""
        triggers = self.wf.get("on", self.wf.get(True, {}))
        assert "pull_request" in triggers

    def test_has_workflow_dispatch(self):
        """ci.yml must allow manual dispatch for on-demand runs."""
        triggers = self.wf.get("on", self.wf.get(True, {}))
        assert "workflow_dispatch" in triggers

    def test_has_detect_changes_job(self):
        """ci.yml must have a detect-changes job using dorny/paths-filter."""
        jobs = self.wf.get("jobs", {})
        assert "detect-changes" in jobs, (
            "ci.yml must have a 'detect-changes' job for smart path filtering"
        )

    def test_detect_changes_uses_paths_filter(self):
        """detect-changes job must use dorny/paths-filter action."""
        job = self.wf["jobs"]["detect-changes"]
        steps = job.get("steps", [])
        uses_list = [s.get("uses", "") for s in steps]
        assert any("dorny/paths-filter" in u for u in uses_list), (
            "detect-changes must use dorny/paths-filter@v3"
        )

    def test_detect_changes_has_outputs(self):
        """detect-changes must expose change-type outputs."""
        job = self.wf["jobs"]["detect-changes"]
        outputs = job.get("outputs", {})
        assert "python" in outputs, "detect-changes must output 'python'"
        assert "tests" in outputs, "detect-changes must output 'tests'"

    def test_test_jobs_have_conditional(self):
        """test-fast and test-integration must have if: conditions."""
        jobs = self.wf.get("jobs", {})
        for job_name in ["test-fast", "test-integration"]:
            if job_name in jobs:
                job = jobs[job_name]
                has_if = "if" in job
                has_needs_detect = "detect-changes" in job.get("needs", [])
                assert has_if or has_needs_detect, (
                    f"{job_name} must have 'if:' condition or depend on detect-changes"
                )


# ============================================================================
# deploy-docs.yml — must exist with correct Pages permissions
# ============================================================================


class TestDeployDocsWorkflow:
    """Validate deploy-docs.yml exists and has correct structure."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.wf = _load_workflow("deploy-docs.yml")

    def test_exists(self):
        """deploy-docs.yml must exist."""
        assert (WORKFLOWS_DIR / "deploy-docs.yml").exists()

    def test_has_pages_permissions(self):
        """deploy-docs.yml must have pages write + id-token write permissions."""
        perms = self.wf.get("permissions", {})
        assert perms.get("pages") == "write", "Must have pages: write permission"
        assert perms.get("id-token") == "write", "Must have id-token: write permission"

    def test_has_build_job(self):
        """deploy-docs.yml must have a build job."""
        assert "build" in self.wf.get("jobs", {})

    def test_has_deploy_job(self):
        """deploy-docs.yml must have a deploy job that needs build."""
        jobs = self.wf.get("jobs", {})
        assert "deploy" in jobs
        deploy = jobs["deploy"]
        needs = deploy.get("needs", [])
        if isinstance(needs, str):
            needs = [needs]
        assert "build" in needs, "deploy must depend on build"

    def test_triggers_on_main_push(self):
        """deploy-docs.yml should trigger on push to main (deployment)."""
        triggers = self.wf.get("on", self.wf.get(True, {}))
        push = triggers.get("push", {})
        branches = push.get("branches", [])
        assert "main" in branches

    def test_uses_upload_pages_artifact(self):
        """Build job must use actions/upload-pages-artifact."""
        build = self.wf["jobs"]["build"]
        uses_list = [s.get("uses", "") for s in build.get("steps", [])]
        assert any("upload-pages-artifact" in u for u in uses_list)

    def test_uses_deploy_pages(self):
        """Deploy job must use actions/deploy-pages."""
        deploy = self.wf["jobs"]["deploy"]
        uses_list = [s.get("uses", "") for s in deploy.get("steps", [])]
        assert any("deploy-pages" in u for u in uses_list)


# ============================================================================
# config-integrity.yml — must NOT have push trigger
# ============================================================================


class TestConfigIntegrityWorkflow:
    """Validate config-integrity.yml triggers."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.wf = _load_workflow("config-integrity.yml")

    def test_no_push_trigger(self):
        """config-integrity.yml must NOT trigger on push (PR-only)."""
        triggers = self.wf.get("on", self.wf.get(True, {}))
        assert "push" not in triggers, (
            "config-integrity.yml should NOT have a push trigger. "
            "Use pull_request only."
        )

    def test_has_pull_request_trigger(self):
        """config-integrity.yml must trigger on pull_request."""
        triggers = self.wf.get("on", self.wf.get(True, {}))
        assert "pull_request" in triggers


# ============================================================================
# General — no workflow should run everything unconditionally
# ============================================================================


class TestNoUnconditionalWorkflows:
    """Ensure no workflow runs all jobs unconditionally on every push."""

    def test_ci_not_unconditional_on_push(self):
        """ci.yml must not trigger all jobs on every push."""
        wf = _load_workflow("ci.yml")
        triggers = wf.get("on", wf.get(True, {}))
        # If push exists, it must have path filters
        if "push" in triggers:
            push = triggers["push"]
            has_paths = "paths" in push or "paths-ignore" in push
            assert has_paths, (
                "If ci.yml has a push trigger, it must have path filtering"
            )
