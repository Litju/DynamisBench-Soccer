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
    variant = tmp_path / "variant.json"
    variant.write_text(policy.model_dump_json(indent=2))
    args = ["timing", command, str(stem), "--policy", str(config), "--sensitivity", str(variant)]
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
        assert report["policy_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
        [sensitivity] = report["sensitivity"]
        assert sensitivity["policy"] == report["policy"]
        assert sensitivity["metrics"] == report["metrics"]
        assert sensitivity["policy_sha256"] == hashlib.sha256(variant.read_bytes()).hexdigest()
        assert sensitivity["policy_sha256"] != report["policy_sha256"]
        assert (
            report["sources"][0]["validated_metadata_sha256"]
            == hashlib.sha256(sidecar.read_bytes()).hexdigest()
        )
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
    assert (
        revised["sources"][0]["validated_metadata_sha256"]
        != report["sources"][0]["validated_metadata_sha256"]
    )

    config.write_bytes(config.read_bytes() + b"\n")
    assert main(args) == 0
    new_policy = json.loads(capsys.readouterr().out)
    assert new_policy["metrics"] == revised["metrics"]
    assert new_policy["policy"] == revised["policy"]
    assert new_policy["policy_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert new_policy["policy_sha256"] != revised["policy_sha256"]

    changed_meta = source.meta.model_copy(update={"bundle_id": source.meta.bundle_id + "-metadata"})
    sidecar.write_text(io.meta_json(changed_meta))
    assert main(args) == 0
    metadata_only = json.loads(capsys.readouterr().out)["sources"][0]
    assert metadata_only["data_sha256"] == new_hash
    assert metadata_only["bundle_id"] == changed_meta.bundle_id
    assert (
        metadata_only["validated_metadata_sha256"]
        == hashlib.sha256(sidecar.read_bytes()).hexdigest()
    )
    assert (
        metadata_only["validated_metadata_sha256"]
        != revised["sources"][0]["validated_metadata_sha256"]
    )

    metadata_bytes = sidecar.read_bytes()
    sidecar.write_text("{")
    assert main(args) == 1
    assert not capsys.readouterr().out
    sidecar.write_bytes(metadata_bytes)
    parquet.write_bytes(parquet.read_bytes() + b"altered")
    assert main(args) == 1
    refusal = capsys.readouterr()
    assert not refusal.out
    assert "sha256 mismatch" in refusal.err
