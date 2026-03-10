"""Infrastructure drift detection for TradeMaster.

Compares actual infrastructure state against declared state (IaC),
detects manual configuration changes, outdated dependencies, and
environment variable mismatches. Tracks drift history and produces
compliance reports with remediation suggestions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DriftCategory(str, Enum):
    MANUAL_CONFIG = "manual_config"
    OUTDATED_DEPENDENCY = "outdated_dependency"
    ENV_VAR_MISMATCH = "env_var_mismatch"
    MISSING_RESOURCE = "missing_resource"
    EXTRA_RESOURCE = "extra_resource"
    CONFIG_VALUE_CHANGED = "config_value_changed"
    SECURITY_DRIFT = "security_drift"


class DriftSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DriftItem:
    """A single detected drift between declared and actual state."""

    category: DriftCategory
    severity: DriftSeverity
    resource: str
    declared_value: str | None
    actual_value: str | None
    message: str
    remediation: str
    detected_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    source_file: str = ""
    auto_remediable: bool = False

    @property
    def drift_id(self) -> str:
        """Stable identifier for deduplication across scans."""
        raw = f"{self.category.value}:{self.resource}:{self.declared_value}:{self.actual_value}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "drift_id": self.drift_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "resource": self.resource,
            "declared_value": self.declared_value,
            "actual_value": self.actual_value,
            "message": self.message,
            "remediation": self.remediation,
            "detected_at": self.detected_at.isoformat(),
            "source_file": self.source_file,
            "auto_remediable": self.auto_remediable,
        }


@dataclass
class ComplianceReport:
    """Aggregated compliance report from a drift detection scan."""

    scan_id: str
    scanned_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    drift_items: list[DriftItem] = field(default_factory=list)
    total_resources_checked: int = 0
    compliant_resources: int = 0
    project_root: str = ""

    @property
    def compliance_score(self) -> float:
        """Compliance percentage (0-100). 100 = no drift detected."""
        if self.total_resources_checked == 0:
            return 100.0
        # Weight by severity
        penalty = 0.0
        for item in self.drift_items:
            penalty += _SEVERITY_WEIGHTS.get(item.severity, 1.0)
        max_penalty = self.total_resources_checked * _SEVERITY_WEIGHTS[DriftSeverity.CRITICAL]
        raw = max(0.0, 1.0 - (penalty / max_penalty))
        return round(raw * 100, 1)

    @property
    def critical_count(self) -> int:
        return sum(1 for d in self.drift_items if d.severity == DriftSeverity.CRITICAL)

    @property
    def error_count(self) -> int:
        return sum(1 for d in self.drift_items if d.severity == DriftSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for d in self.drift_items if d.severity == DriftSeverity.WARNING)

    @property
    def auto_remediable_count(self) -> int:
        return sum(1 for d in self.drift_items if d.auto_remediable)

    def summary(self) -> dict[str, Any]:
        return {
            "scan_id": self.scan_id,
            "scanned_at": self.scanned_at.isoformat(),
            "compliance_score": self.compliance_score,
            "total_resources_checked": self.total_resources_checked,
            "compliant_resources": self.compliant_resources,
            "drift_count": len(self.drift_items),
            "by_severity": {
                "critical": self.critical_count,
                "error": self.error_count,
                "warning": self.warning_count,
                "info": sum(1 for d in self.drift_items if d.severity == DriftSeverity.INFO),
            },
            "auto_remediable": self.auto_remediable_count,
        }

    def to_dict(self) -> dict[str, Any]:
        result = self.summary()
        result["drift_items"] = [item.to_dict() for item in self.drift_items]
        return result


_SEVERITY_WEIGHTS: dict[DriftSeverity, float] = {
    DriftSeverity.INFO: 0.1,
    DriftSeverity.WARNING: 0.3,
    DriftSeverity.ERROR: 0.7,
    DriftSeverity.CRITICAL: 1.0,
}


@dataclass
class _DriftHistoryEntry:
    """Internal record for tracking drift over time."""

    scan_id: str
    scanned_at: datetime
    compliance_score: float
    drift_count: int
    drift_ids: set[str] = field(default_factory=set)


class InfrastructureDriftDetector:
    """Detects and tracks infrastructure drift for TradeMaster.

    Compares the actual state of the running environment against the
    declared state in IaC files (docker-compose, Dockerfiles, Terraform,
    env templates). Produces compliance reports and remediation suggestions.
    """

    def __init__(
        self,
        project_root: str | Path,
        *,
        env_template_file: str = ".env.example",
        docker_compose_file: str = "docker-compose.yml",
        history_limit: int = 100,
    ):
        self.project_root = Path(project_root)
        self.env_template_file = env_template_file
        self.docker_compose_file = docker_compose_file
        self._history: list[_DriftHistoryEntry] = []
        self._history_limit = history_limit
        self._scan_counter = 0

    # -- Public API --

    def full_scan(self) -> ComplianceReport:
        """Run a comprehensive drift detection scan.

        Checks: environment variables, Docker configuration, dependency
        versions, file permissions, and infrastructure configuration.
        """
        self._scan_counter += 1
        scan_id = (
            f"scan-{self._scan_counter}-"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
        )

        report = ComplianceReport(
            scan_id=scan_id,
            project_root=str(self.project_root),
        )

        # Run all detection passes
        checkers = [
            self._check_env_vars,
            self._check_docker_compose_drift,
            self._check_dependency_versions,
            self._check_dockerfile_drift,
            self._check_terraform_drift,
            self._check_security_configs,
        ]

        for checker in checkers:
            try:
                items, checked = checker()
                report.drift_items.extend(items)
                report.total_resources_checked += checked
            except Exception:
                logger.exception("Drift check failed: %s", checker.__name__)

        report.compliant_resources = (
            report.total_resources_checked - len(report.drift_items)
        )

        # Record in history
        self._record_history(report)

        logger.info(
            "Drift scan complete: score=%.1f%%, drifts=%d",
            report.compliance_score,
            len(report.drift_items),
        )
        return report

    def check_env_vars(self) -> list[DriftItem]:
        """Check environment variables against the template."""
        items, _ = self._check_env_vars()
        return items

    def check_dependencies(self) -> list[DriftItem]:
        """Check for outdated or mismatched dependencies."""
        items, _ = self._check_dependency_versions()
        return items

    def get_history(self) -> list[dict[str, Any]]:
        """Return drift scan history for trend analysis."""
        return [
            {
                "scan_id": entry.scan_id,
                "scanned_at": entry.scanned_at.isoformat(),
                "compliance_score": entry.compliance_score,
                "drift_count": entry.drift_count,
            }
            for entry in self._history
        ]

    def identify_recurring_drifts(self) -> list[dict[str, Any]]:
        """Identify drift items that appear across multiple scans.

        Useful for finding systemic issues vs. one-off misconfigurations.
        """
        if len(self._history) < 2:
            return []

        # Count how many scans each drift_id appears in
        drift_frequency: dict[str, int] = {}
        for entry in self._history:
            for drift_id in entry.drift_ids:
                drift_frequency[drift_id] = drift_frequency.get(drift_id, 0) + 1

        total_scans = len(self._history)
        recurring = [
            {
                "drift_id": did,
                "occurrences": count,
                "total_scans": total_scans,
                "recurrence_rate": round(count / total_scans, 2),
                "pattern": "persistent" if count == total_scans else "intermittent",
            }
            for did, count in drift_frequency.items()
            if count >= 2
        ]

        recurring.sort(key=lambda r: r["occurrences"], reverse=True)
        return recurring

    def get_remediation_plan(
        self,
        report: ComplianceReport,
        *,
        auto_only: bool = False,
    ) -> list[dict[str, str]]:
        """Generate an ordered remediation plan from a compliance report.

        Returns actions sorted by severity (critical first). If auto_only
        is True, only includes items that can be auto-remediated.
        """
        items = report.drift_items
        if auto_only:
            items = [i for i in items if i.auto_remediable]

        severity_order = {
            DriftSeverity.CRITICAL: 0,
            DriftSeverity.ERROR: 1,
            DriftSeverity.WARNING: 2,
            DriftSeverity.INFO: 3,
        }
        items_sorted = sorted(items, key=lambda i: severity_order.get(i.severity, 9))

        return [
            {
                "step": str(idx + 1),
                "severity": item.severity.value,
                "resource": item.resource,
                "action": item.remediation,
                "auto_remediable": str(item.auto_remediable),
            }
            for idx, item in enumerate(items_sorted)
        ]

    # -- Private detection methods --

    def _check_env_vars(self) -> tuple[list[DriftItem], int]:
        """Compare .env.example template against actual environment."""
        template_path = self.project_root / self.env_template_file
        items: list[DriftItem] = []
        checked = 0

        if not template_path.exists():
            logger.debug("No env template found at %s", template_path)
            return items, checked

        declared_vars = self._parse_env_file(template_path)
        checked = len(declared_vars)

        for var_name, declared_val in declared_vars.items():
            actual_val = os.environ.get(var_name)

            if actual_val is None:
                items.append(
                    DriftItem(
                        category=DriftCategory.ENV_VAR_MISMATCH,
                        severity=self._env_var_severity(var_name),
                        resource=f"env:{var_name}",
                        declared_value=declared_val,
                        actual_value=None,
                        message=f"Environment variable {var_name} declared in template but not set",
                        remediation=f"Set {var_name} in your .env file or environment",
                        source_file=str(template_path),
                        auto_remediable=False,
                    )
                )
            elif declared_val and actual_val != declared_val and not self._is_placeholder(declared_val):
                items.append(
                    DriftItem(
                        category=DriftCategory.ENV_VAR_MISMATCH,
                        severity=DriftSeverity.WARNING,
                        resource=f"env:{var_name}",
                        declared_value=declared_val,
                        actual_value="<redacted>" if self._is_secret_var(var_name) else actual_val,
                        message=f"Environment variable {var_name} differs from template default",
                        remediation=f"Review {var_name} value; update template or environment as needed",
                        source_file=str(template_path),
                        auto_remediable=False,
                    )
                )

        return items, checked

    def _check_docker_compose_drift(self) -> tuple[list[DriftItem], int]:
        """Detect drift in Docker Compose configuration."""
        compose_path = self.project_root / self.docker_compose_file
        items: list[DriftItem] = []
        checked = 0

        if not compose_path.exists():
            return items, checked

        try:
            compose_content = compose_path.read_text()
        except OSError:
            logger.warning("Cannot read %s", compose_path)
            return items, checked

        # Check if running containers match declared services
        declared_services = self._extract_compose_services(compose_content)
        checked = len(declared_services)

        running_containers = self._get_running_containers()

        for service in declared_services:
            if service not in running_containers:
                items.append(
                    DriftItem(
                        category=DriftCategory.MISSING_RESOURCE,
                        severity=DriftSeverity.ERROR,
                        resource=f"docker:service:{service}",
                        declared_value=service,
                        actual_value=None,
                        message=f"Service '{service}' declared in docker-compose but not running",
                        remediation=f"Run: docker-compose up -d {service}",
                        source_file=str(compose_path),
                        auto_remediable=True,
                    )
                )

        for container in running_containers:
            if container not in declared_services:
                items.append(
                    DriftItem(
                        category=DriftCategory.EXTRA_RESOURCE,
                        severity=DriftSeverity.WARNING,
                        resource=f"docker:container:{container}",
                        declared_value=None,
                        actual_value=container,
                        message=f"Container '{container}' running but not declared in docker-compose",
                        remediation=f"Remove undeclared container or add to docker-compose.yml",
                        source_file=str(compose_path),
                        auto_remediable=False,
                    )
                )

        # Check for image version pinning
        unpinned = self._find_unpinned_images(compose_content)
        checked += len(unpinned)
        for image in unpinned:
            items.append(
                DriftItem(
                    category=DriftCategory.SECURITY_DRIFT,
                    severity=DriftSeverity.WARNING,
                    resource=f"docker:image:{image}",
                    declared_value=image,
                    actual_value=None,
                    message=f"Image '{image}' uses 'latest' tag or no tag; pin to specific version",
                    remediation=f"Pin {image} to a specific version tag in docker-compose.yml",
                    source_file=str(compose_path),
                    auto_remediable=False,
                )
            )

        return items, checked

    def _check_dependency_versions(self) -> tuple[list[DriftItem], int]:
        """Check Python and Node.js dependency freshness."""
        items: list[DriftItem] = []
        checked = 0

        # Python: check requirements.txt or pyproject.toml
        req_files = [
            self.project_root / "backend" / "requirements.txt",
            self.project_root / "backend" / "pyproject.toml",
        ]

        for req_file in req_files:
            if not req_file.exists():
                continue
            try:
                content = req_file.read_text()
            except OSError:
                continue

            unpinned = self._find_unpinned_python_deps(content, req_file.name)
            checked += len(unpinned)
            for dep_name in unpinned:
                items.append(
                    DriftItem(
                        category=DriftCategory.OUTDATED_DEPENDENCY,
                        severity=DriftSeverity.WARNING,
                        resource=f"python:dep:{dep_name}",
                        declared_value=dep_name,
                        actual_value=None,
                        message=f"Python dependency '{dep_name}' is not version-pinned",
                        remediation=f"Pin '{dep_name}' to a specific version in {req_file.name}",
                        source_file=str(req_file),
                        auto_remediable=False,
                    )
                )

        # Node.js: check for lockfile existence
        frontend_dir = self.project_root / "frontend"
        if frontend_dir.exists():
            pkg_json = frontend_dir / "package.json"
            lock_files = [
                frontend_dir / "package-lock.json",
                frontend_dir / "yarn.lock",
                frontend_dir / "pnpm-lock.yaml",
            ]
            if pkg_json.exists():
                checked += 1
                if not any(lf.exists() for lf in lock_files):
                    items.append(
                        DriftItem(
                            category=DriftCategory.OUTDATED_DEPENDENCY,
                            severity=DriftSeverity.ERROR,
                            resource="node:lockfile",
                            declared_value="package-lock.json or equivalent",
                            actual_value=None,
                            message="No lockfile found for frontend dependencies",
                            remediation="Run 'npm install' or equivalent to generate a lockfile",
                            source_file=str(pkg_json),
                            auto_remediable=True,
                        )
                    )

        return items, checked

    def _check_dockerfile_drift(self) -> tuple[list[DriftItem], int]:
        """Check Dockerfiles for best-practice deviations."""
        items: list[DriftItem] = []
        checked = 0

        docker_dir = self.project_root / "infrastructure" / "docker"
        if not docker_dir.exists():
            return items, checked

        for dockerfile in docker_dir.glob("*.Dockerfile"):
            checked += 1
            try:
                content = dockerfile.read_text()
            except OSError:
                continue

            # Check for running as root
            if "USER" not in content:
                items.append(
                    DriftItem(
                        category=DriftCategory.SECURITY_DRIFT,
                        severity=DriftSeverity.ERROR,
                        resource=f"docker:file:{dockerfile.name}",
                        declared_value="non-root USER",
                        actual_value="root (default)",
                        message=f"{dockerfile.name} runs as root; specify a non-root USER",
                        remediation=f"Add 'USER nonroot' to {dockerfile.name}",
                        source_file=str(dockerfile),
                        auto_remediable=False,
                    )
                )

            # Check for COPY vs ADD
            if re.search(r"^\s*ADD\s+(?!https?://)", content, re.MULTILINE):
                items.append(
                    DriftItem(
                        category=DriftCategory.SECURITY_DRIFT,
                        severity=DriftSeverity.WARNING,
                        resource=f"docker:file:{dockerfile.name}",
                        declared_value="COPY",
                        actual_value="ADD",
                        message=f"{dockerfile.name} uses ADD for local files; prefer COPY",
                        remediation=f"Replace ADD with COPY in {dockerfile.name} for local files",
                        source_file=str(dockerfile),
                        auto_remediable=True,
                    )
                )

            # Check base image pinning
            from_lines = re.findall(r"^FROM\s+(\S+)", content, re.MULTILINE)
            for image in from_lines:
                if ":" not in image or image.endswith(":latest"):
                    items.append(
                        DriftItem(
                            category=DriftCategory.SECURITY_DRIFT,
                            severity=DriftSeverity.WARNING,
                            resource=f"docker:base_image:{image}",
                            declared_value="pinned version",
                            actual_value=image,
                            message=f"Base image '{image}' in {dockerfile.name} is not version-pinned",
                            remediation=f"Pin '{image}' to a specific digest or version tag",
                            source_file=str(dockerfile),
                            auto_remediable=False,
                        )
                    )

        return items, checked

    def _check_terraform_drift(self) -> tuple[list[DriftItem], int]:
        """Check for Terraform state drift if terraform files exist."""
        items: list[DriftItem] = []
        checked = 0

        tf_dir = self.project_root / "infrastructure" / "terraform"
        if not tf_dir.exists():
            return items, checked

        tf_files = list(tf_dir.glob("*.tf"))
        checked = len(tf_files)

        # Check for .terraform.lock.hcl
        lock_file = tf_dir / ".terraform.lock.hcl"
        if tf_files and not lock_file.exists():
            items.append(
                DriftItem(
                    category=DriftCategory.MISSING_RESOURCE,
                    severity=DriftSeverity.ERROR,
                    resource="terraform:lockfile",
                    declared_value=".terraform.lock.hcl",
                    actual_value=None,
                    message="Terraform dependency lock file missing",
                    remediation="Run 'terraform init' to generate .terraform.lock.hcl",
                    source_file=str(tf_dir),
                    auto_remediable=True,
                )
            )

        # Check for backend configuration (remote state)
        has_backend = False
        for tf_file in tf_files:
            try:
                content = tf_file.read_text()
                if re.search(r'backend\s+"', content):
                    has_backend = True
                    break
            except OSError:
                continue

        if tf_files and not has_backend:
            items.append(
                DriftItem(
                    category=DriftCategory.MANUAL_CONFIG,
                    severity=DriftSeverity.ERROR,
                    resource="terraform:backend",
                    declared_value="remote backend",
                    actual_value="local (default)",
                    message="Terraform uses local state; configure a remote backend for team use",
                    remediation=(
                        "Add a backend block (e.g., S3, GCS, or Terraform Cloud) "
                        "to your Terraform configuration"
                    ),
                    source_file=str(tf_dir),
                    auto_remediable=False,
                )
            )

        return items, checked

    def _check_security_configs(self) -> tuple[list[DriftItem], int]:
        """Check for security-related configuration drift."""
        items: list[DriftItem] = []
        checked = 0

        # Check .env file is in .gitignore
        gitignore = self.project_root / ".gitignore"
        checked += 1
        if gitignore.exists():
            try:
                content = gitignore.read_text()
                if ".env" not in content:
                    items.append(
                        DriftItem(
                            category=DriftCategory.SECURITY_DRIFT,
                            severity=DriftSeverity.CRITICAL,
                            resource="git:gitignore:.env",
                            declared_value=".env in .gitignore",
                            actual_value=".env NOT in .gitignore",
                            message=".env file is not listed in .gitignore; secrets may be committed",
                            remediation="Add '.env' and '.env.*' to .gitignore immediately",
                            source_file=str(gitignore),
                            auto_remediable=True,
                        )
                    )
            except OSError:
                pass
        else:
            items.append(
                DriftItem(
                    category=DriftCategory.SECURITY_DRIFT,
                    severity=DriftSeverity.CRITICAL,
                    resource="git:gitignore",
                    declared_value=".gitignore file",
                    actual_value=None,
                    message="No .gitignore file found; secrets may be committed to version control",
                    remediation="Create a .gitignore file with standard exclusions",
                    source_file=str(self.project_root),
                    auto_remediable=True,
                )
            )

        # Check if any .env files are tracked by git
        checked += 1
        tracked_env_files = self._get_git_tracked_env_files()
        for env_file in tracked_env_files:
            items.append(
                DriftItem(
                    category=DriftCategory.SECURITY_DRIFT,
                    severity=DriftSeverity.CRITICAL,
                    resource=f"git:tracked:{env_file}",
                    declared_value="untracked",
                    actual_value="tracked in git",
                    message=f"Secret file '{env_file}' is tracked by git",
                    remediation=(
                        f"Run: git rm --cached {env_file} && "
                        f"echo '{env_file}' >> .gitignore"
                    ),
                    source_file=env_file,
                    auto_remediable=False,
                )
            )

        return items, checked

    # -- Utility methods --

    def _record_history(self, report: ComplianceReport) -> None:
        """Record scan result in drift history."""
        entry = _DriftHistoryEntry(
            scan_id=report.scan_id,
            scanned_at=report.scanned_at,
            compliance_score=report.compliance_score,
            drift_count=len(report.drift_items),
            drift_ids={item.drift_id for item in report.drift_items},
        )
        self._history.append(entry)
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit:]

    @staticmethod
    def _parse_env_file(path: Path) -> dict[str, str]:
        """Parse a .env-style file into a dict of name -> value."""
        result: dict[str, str] = {}
        try:
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("\"'")
                    result[key] = value
        except OSError:
            logger.warning("Cannot read env file: %s", path)
        return result

    @staticmethod
    def _is_placeholder(value: str) -> bool:
        """Check if an env value is a placeholder (not a real default)."""
        placeholders = {
            "", "changeme", "your-value-here", "xxx", "TODO",
            "CHANGE_ME", "replace_me", "your_secret_here",
        }
        return value.lower() in {p.lower() for p in placeholders} or value.startswith("<")

    @staticmethod
    def _is_secret_var(name: str) -> bool:
        """Check if a variable name likely contains a secret."""
        secret_patterns = {
            "password", "secret", "key", "token", "api_key",
            "private", "credential", "auth",
        }
        name_lower = name.lower()
        return any(pattern in name_lower for pattern in secret_patterns)

    @staticmethod
    def _env_var_severity(var_name: str) -> DriftSeverity:
        """Determine severity of a missing env var based on its name."""
        critical_patterns = {"database_url", "secret_key", "jwt_secret", "db_password"}
        error_patterns = {"redis_url", "api_key", "postgres_"}
        name_lower = var_name.lower()

        if any(p in name_lower for p in critical_patterns):
            return DriftSeverity.CRITICAL
        if any(p in name_lower for p in error_patterns):
            return DriftSeverity.ERROR
        return DriftSeverity.WARNING

    @staticmethod
    def _extract_compose_services(content: str) -> list[str]:
        """Extract service names from docker-compose YAML content."""
        services: list[str] = []
        in_services = False
        indent_level = -1

        for line in content.splitlines():
            stripped = line.strip()
            if stripped == "services:":
                in_services = True
                continue

            if in_services:
                if stripped and not stripped.startswith("#"):
                    current_indent = len(line) - len(line.lstrip())
                    if indent_level < 0:
                        indent_level = current_indent

                    if current_indent == indent_level and stripped.endswith(":"):
                        services.append(stripped.rstrip(":"))
                    elif current_indent == 0 and not stripped.startswith(" "):
                        # Left top-level services block
                        break

        return services

    @staticmethod
    def _find_unpinned_images(compose_content: str) -> list[str]:
        """Find Docker images in compose that lack version pinning."""
        unpinned: list[str] = []
        for match in re.finditer(r"image:\s*(\S+)", compose_content):
            image = match.group(1).strip("\"'")
            if ":" not in image or image.endswith(":latest"):
                unpinned.append(image)
        return unpinned

    @staticmethod
    def _find_unpinned_python_deps(content: str, filename: str) -> list[str]:
        """Find Python dependencies without version pinning."""
        unpinned: list[str] = []

        if filename == "requirements.txt":
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("-"):
                    continue
                if "==" not in line and ">=" not in line and "<=" not in line:
                    dep_name = re.split(r"[<>=!~]", line)[0].strip()
                    if dep_name:
                        unpinned.append(dep_name)
        elif filename == "pyproject.toml":
            in_deps = False
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("dependencies") or stripped.startswith("[project.dependencies]"):
                    in_deps = True
                    continue
                if in_deps:
                    if stripped.startswith("[") and "dependencies" not in stripped:
                        break
                    # Match quoted dependency strings without version specifiers
                    dep_match = re.match(r'["\']([a-zA-Z0-9_-]+)["\']', stripped)
                    if dep_match:
                        unpinned.append(dep_match.group(1))

        return unpinned

    def _get_running_containers(self) -> list[str]:
        """Get list of running Docker container names (project-scoped)."""
        try:
            result = subprocess.run(
                ["docker", "compose", "ps", "--format", "{{.Service}}"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.project_root,
            )
            if result.returncode == 0:
                return [s.strip() for s in result.stdout.splitlines() if s.strip()]
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.debug("Could not query running Docker containers")
        return []

    def _get_git_tracked_env_files(self) -> list[str]:
        """Check if any .env files are tracked by git."""
        try:
            result = subprocess.run(
                ["git", "ls-files", "--cached", "*.env", ".env*"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.project_root,
            )
            if result.returncode == 0:
                return [
                    f.strip() for f in result.stdout.splitlines()
                    if f.strip() and not f.strip().endswith(".example")
                ]
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.debug("Could not query git tracked files")
        return []
