"""Research approval alone never activates or authorizes Testnet."""

from app.models.research_experiment import ResearchExperiment
from app.services.research.testnet_release_gate import evaluate_testnet_eligibility


def experiment(status: str) -> ResearchExperiment:
    return ResearchExperiment(
        id="experiment",
        name="candidate",
        status=status,
        code_revision="a" * 40,
        protocol_sha256="b" * 64,
        product_json="{}",
        cost_profile_json="{}",
        approval_gate_json="{}",
    )


def test_approved_research_still_needs_separate_testnet_release() -> None:
    result = evaluate_testnet_eligibility(
        experiment("APPROVED"),
        prospective_days=30,
        unresolved_failures=0,
        explicit_testnet_release=False,
    )
    assert result.eligible is False
    assert result.order_submission_allowed is False
    assert result.execution_authorization == "none"


def test_eligibility_is_metadata_not_execution_authorization() -> None:
    result = evaluate_testnet_eligibility(
        experiment("APPROVED"),
        prospective_days=30,
        unresolved_failures=0,
        explicit_testnet_release=True,
    )
    assert result.eligible is True
    assert result.order_submission_allowed is False
    assert result.execution_authorization == "none"
