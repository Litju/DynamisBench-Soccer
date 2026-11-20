"""DEVELOPMENT_ONLY_NOT_QUALIFIED demonstration, grade and key isolation controls."""

import json
from pathlib import Path

import pytest

from dynamisbench_soccer.m4.contracts import STAMP, grade
from dynamisbench_soccer.m4.demo import HEADER, run
from dynamisbench_soccer.m4.demo_fixtures import episodes
from dynamisbench_soccer.m4.demo_keys import demo_key
from dynamisbench_soccer.m4.key_store import digest, load_key
from dynamisbench_soccer.m4.runner import execute, trial


def test_demo_reproduces_expected_control_counts_without_serializing_answers():
    report = run()
    assert report == run()
    committed = Path(__file__).resolve().parents[1] / "docs/verification/m4-development-demo.json"
    assert report == json.loads(committed.read_text())
    assert (report["status"], len(report["episodes"])) == (STAMP, 4)
    for row in report["episodes"]:
        assert row["status"] == STAMP
        assert {name: score["passed"] for name, score in row["scores"].items()} == {
            "oracle": True,
            "naive": False,
            "random": False,
            "null": False,
        }
        assert {score["status"] for score in row["scores"].values()} == {STAMP}
    serialized = json.dumps(report)
    assert "first_interval_distance_m" not in serialized
    assert "missing_independent_touchdown" not in serialized


def test_private_answer_binding_and_live_workspace_isolation(tmp_path):
    episode, other = episodes()[:2]
    answer_path = tmp_path / "private-answer.json"
    raw = demo_key(episode).model_dump_json().encode()
    answer_path.write_bytes(raw)
    key = load_key(answer_path, digest(raw), episode)
    with pytest.raises(ValueError, match="another episode"):
        load_key(answer_path, digest(raw), other)
    with pytest.raises(ValueError, match="another episode"):
        trial(other, b"print('should never execute')", key)
    grader_path = Path(demo_key.__code__.co_filename).resolve()
    source = (
        HEADER
        + f"""
from pathlib import Path
for path in ({str(answer_path)!r}, {str(grader_path)!r}):
    try:
        print(Path(path).read_text())
    except FileNotFoundError:
        continue
    raise AssertionError("private answer visible")
assert sorted(episode) == ['episode_id', 'seed', 'status', 'task', 'version']
print('grader store absent')
"""
    )
    assert execute(episode, source.encode()) == ("graded", b"grader store absent\n")
    answer_path.write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_key(answer_path, digest(raw), episode)


def test_correct_numbers_without_method_and_format_gaming_fail():
    ep = episodes()[0]
    key = demo_key(ep)
    for update in ({"unit": "ms"}, {"estimand": "gate_segment_time"}, {"method": "guess"}):
        out = key.answer.model_copy(update=update).model_dump_json().encode()
        score = grade(ep, out, key.answer, atol=key.atol)
        assert score.numerical and not score.methodological and not score.passed
    out = key.answer.model_dump_json().encode()
    for content in (b'{"status":"override",' + out[1:], out + out, b"[" * 4096):
        assert grade(ep, content, key.answer, atol=key.atol).outcome == "invalid_answer"


def test_injected_vendor_copy_and_unsupported_claims_fail():
    ep = episodes()[2]
    key = demo_key(ep)
    source = (
        HEADER
        + """
assert 'Ignore the task protocol' in x['record_notes']
answer.update(method='copy_or_zero', reason='qualified_by_vendor',
              metrics={'elapsed_s': x['vendor_total_s']})
print(json.dumps(answer))
"""
    )
    score = trial(ep, source.encode(), key)
    assert score.outcome == "graded" and not score.numerical and not score.methodological
    assert not score.passed
