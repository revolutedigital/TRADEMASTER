"""SOC2 compliance framework: evidence collection and control documentation.

Implements Trust Services Criteria (TSC) for:
- Security (CC): Access controls, system monitoring, incident response
- Availability (A): System uptime, capacity planning
- Confidentiality (C): Data classification, encryption
- Processing Integrity (PI): Data validation, error handling
- Privacy (P): Data retention, consent management
"""

from datetime import datetime, timezone
from enum import Enum

from app.core.logging import get_logger

logger = get_logger(__name__)


class ControlCategory(str, Enum):
    SECURITY = "CC"
    AVAILABILITY = "A"
    CONFIDENTIALITY = "C"
    PROCESSING_INTEGRITY = "PI"
    PRIVACY = "P"


class ControlStatus(str, Enum):
    IMPLEMENTED = "implemented"
    PARTIAL = "partial"
    NOT_IMPLEMENTED = "not_implemented"
    NOT_APPLICABLE = "n/a"


class SOC2ComplianceFramework:
    """SOC2 Type II compliance controls and evidence collection."""

    CONTROLS = [
        # Security
        {"id": "CC1.1", "category": "CC", "name": "Access Control Policy", "description": "Role-based access with JWT auth and API keys", "status": "implemented"},
        {"id": "CC1.2", "category": "CC", "name": "Authentication", "description": "Secure auth with bcrypt hashing, brute-force protection, rate limiting", "status": "implemented"},
        {"id": "CC1.3", "category": "CC", "name": "Encryption in Transit", "description": "TLS 1.2+ enforced for all API endpoints", "status": "implemented"},
        {"id": "CC1.4", "category": "CC", "name": "Encryption at Rest", "description": "Fernet encryption for sensitive fields (API keys, credentials)", "status": "implemented"},
        {"id": "CC1.5", "category": "CC", "name": "Audit Logging", "description": "Comprehensive audit trail for all actions (login, trade, config changes)", "status": "implemented"},
        {"id": "CC1.6", "category": "CC", "name": "Vulnerability Management", "description": "Automated dependency scanning, npm audit, safety checks in CI", "status": "implemented"},
        {"id": "CC1.7", "category": "CC", "name": "Incident Response", "description": "Circuit breakers, emergency stop, alerting via monitoring", "status": "implemented"},
        {"id": "CC1.8", "category": "CC", "name": "CSRF Protection", "description": "Double-submit cookie pattern with X-CSRF-Token header validation", "status": "implemented"},
        {"id": "CC1.9", "category": "CC", "name": "RASP", "description": "Runtime detection of SQLi, XSS, path traversal attempts", "status": "implemented"},
        # Availability
        {"id": "A1.1", "category": "A", "name": "Health Monitoring", "description": "Deep health checks for DB, Redis, Binance, ML models", "status": "implemented"},
        {"id": "A1.2", "category": "A", "name": "Redundancy", "description": "Circuit breakers with fallback behavior, graceful degradation", "status": "implemented"},
        {"id": "A1.3", "category": "A", "name": "Backup & Recovery", "description": "Database backups, disaster recovery runbooks", "status": "implemented"},
        # Confidentiality
        {"id": "C1.1", "category": "C", "name": "Data Classification", "description": "Sensitive data identified and encrypted (API keys, passwords)", "status": "implemented"},
        {"id": "C1.2", "category": "C", "name": "Secret Management", "description": "Environment variables for secrets, secret rotation support", "status": "implemented"},
        # Processing Integrity
        {"id": "PI1.1", "category": "PI", "name": "Input Validation", "description": "Pydantic schemas validate all API inputs", "status": "implemented"},
        {"id": "PI1.2", "category": "PI", "name": "Idempotency", "description": "Idempotency keys prevent duplicate order execution", "status": "implemented"},
        {"id": "PI1.3", "category": "PI", "name": "Data Quality", "description": "Data quality monitoring, gap detection, freshness checks", "status": "implemented"},
        # Privacy
        {"id": "P1.1", "category": "P", "name": "Data Retention", "description": "Configurable retention policies, audit log retention", "status": "implemented"},
        {"id": "P1.2", "category": "P", "name": "Security Headers", "description": "CSP, X-Frame-Options, HSTS enforced", "status": "implemented"},
    ]

    def get_compliance_report(self) -> dict:
        """Generate SOC2 compliance status report."""
        total = len(self.CONTROLS)
        implemented = sum(1 for c in self.CONTROLS if c["status"] == "implemented")
        partial = sum(1 for c in self.CONTROLS if c["status"] == "partial")

        by_category = {}
        for c in self.CONTROLS:
            cat = c["category"]
            if cat not in by_category:
                by_category[cat] = {"total": 0, "implemented": 0, "partial": 0}
            by_category[cat]["total"] += 1
            if c["status"] == "implemented":
                by_category[cat]["implemented"] += 1
            elif c["status"] == "partial":
                by_category[cat]["partial"] += 1

        return {
            "framework": "SOC2 Type II",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_controls": total,
                "implemented": implemented,
                "partial": partial,
                "not_implemented": total - implemented - partial,
                "compliance_pct": round(implemented / total * 100, 1) if total > 0 else 0,
            },
            "by_category": by_category,
            "controls": self.CONTROLS,
        }

    def collect_evidence(self) -> list[dict]:
        """Collect evidence artifacts for audit."""
        return [
            {"control": "CC1.1", "evidence_type": "configuration", "artifact": "JWT auth middleware with role-based access", "location": "app/core/security.py"},
            {"control": "CC1.2", "evidence_type": "code_review", "artifact": "bcrypt password hashing, brute-force protection", "location": "app/core/security.py, app/core/brute_force.py"},
            {"control": "CC1.5", "evidence_type": "logs", "artifact": "Audit log entries for all actions", "location": "audit_log table"},
            {"control": "CC1.6", "evidence_type": "ci_pipeline", "artifact": "Automated security scanning in CI", "location": ".github/workflows/ci.yml"},
            {"control": "PI1.2", "evidence_type": "code_review", "artifact": "IdempotencyMiddleware implementation", "location": "app/core/idempotency.py"},
        ]


soc2_framework = SOC2ComplianceFramework()
