"""Protocol guard checks for the managed executed-flow pilot."""

from scripts.research.fx_fast12_registry import PREREGISTRATION, REGISTRY
from scripts.research.fx_fast5_registry import problems, read_events, sha256_of


def test_fast12_registry_is_valid() -> None:
    assert problems(read_events(REGISTRY), sha256_of(PREREGISTRATION)) == []
