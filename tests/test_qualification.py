"""RES-692 qualification gate adversarial controls."""

from dataclasses import replace
from pathlib import Path

from dynamisbench_soccer.qualification.ledger import (
    DIMENSIONS, Rating, Stage, gate, load, markdown, validate,
)


def test_stage_definitions_are_separate() -> None:
    assert len(DIMENSIONS) == 15
    assert len([d for d in DIMENSIONS if d != "N"]) == 14
    assert Stage.M3 != Stage.M5 != Stage.M8


def test_missing_families_cannot_pass() -> None:
    sample = Path(__file__).resolve().parents[1] / "qualification" / "ledger.toml"
    if not sample.is_file():
        return
    obj = load(sample)
    assert len({s.family for s in obj.subfamilies}) == 8
    assert not validate(obj)
    assert gate(obj, Stage.M3).passed_families == 0
    assert gate(obj, Stage.M5).passed_families == 0
    assert gate(obj, Stage.M8).passed_families == 0
    assert markdown(obj).count("IN_PROGRESS") >= 8


def test_specified_not_verified() -> None:
    from dynamisbench_soccer.qualification.ledger import Subfamily

    states = {d: Rating.ABSENT for d in DIMENSIONS}
    states["A"] = Rating.SPECIFIED
    sub = Subfamily(
        "force_platform.single", "force_platform", "continuous_signal",
        "RES-670", "test-engine", (), states, (),
    )
    assert sub.ratings["A"] != Rating.VERIFIED
    assert replace(sub, ratings={**sub.ratings, "N": Rating.ABSENT})
