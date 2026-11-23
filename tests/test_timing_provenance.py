"""Independent event-stream replay and input binding (RES-698)."""

import hashlib
import json
from pathlib import Path

import pytest
from test_timing_contact import policy as contact_policy
from test_timing_gates import gate_policy

from dynamisbench_soccer.cli import main
from dynamisbench_soccer.reference.exporters import export_case
from dynamisbench_soccer.signals import io


@pytest.mark.parametrize(
    ("command", "policy"), [("gates", gate_policy()), ("contact", contact_policy())]
)
def test_event_cli_replays_source_and_refuses_mutation(command, policy, tmp_path: Path, capsys):
    stem = tmp_path / "observation"
    config = tmp_path / "policy.json"
    config.write_text(policy.model_dump_json())
    args = ["timing", command, str(stem), "--policy", str(config)]
    evidence = []
    for _ in range(2):
        source = export_case(f"timing_contact.{command}").observation
        parquet, sidecar = io.write(source, stem)
        assert main(args) == 0
        output = capsys.readouterr().out
        report = json.loads(output)
        assert report["sources"][0]["bundle_id"] == source.meta.bundle_id
        assert (
            report["sources"][0]["data_sha256"] == hashlib.sha256(parquet.read_bytes()).hexdigest()
        )
        assert report["policy"] == policy.model_dump(mode="json")
        evidence.append((output, parquet.read_bytes(), sidecar.read_bytes()))
    assert evidence[0] == evidence[1]

    source["t_s"][1] += 0.002
    io.write(source, stem)
    assert main(args) == 0
    revised = json.loads(capsys.readouterr().out)
    assert revised["metrics"] != report["metrics"]
    new_hash = revised["sources"][0]["data_sha256"]
    assert new_hash == hashlib.sha256(parquet.read_bytes()).hexdigest()
    assert new_hash != report["sources"][0]["data_sha256"]

    parquet.write_bytes(parquet.read_bytes() + b"altered")
    assert main(args) == 1
    refusal = capsys.readouterr()
    assert not refusal.out
    assert "sha256 mismatch" in refusal.err
