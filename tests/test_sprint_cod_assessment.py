"""Sprint F-V and COD assessment versus independent closed forms (RES-673).

Mono-exponential sprint v(t) = Vmax (1 - exp(-t/tau)): F0 = m Vmax / tau, Pmax = F0 Vmax / 4,
F-V slope = -F0 / Vmax, x(t) = Vmax (t + tau (exp(-t/tau) - 1)). COD speed profile is built
piecewise from closed forms in the test.
"""

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from dynamisbench_soccer.assessment import cod, sprint
from dynamisbench_soccer.assessment.cod import CodSpec
from dynamisbench_soccer.assessment.sprint import FvSpec, StartSpec
from dynamisbench_soccer.cli import main
from dynamisbench_soccer.engines.radar_laser import ProcessingOptions, process
from dynamisbench_soccer.engines.radar_laser_acquisition import ChannelStage
from dynamisbench_soccer.reference.exporters import export_case
from dynamisbench_soccer.reference.motion_models import MonoExponentialSprint

VMAX, TAU, MASS = 9.0, 1.2, 80.0
TRUTH = MonoExponentialSprint(VMAX, TAU)
F0, PMAX = MASS * VMAX / TAU, MASS * VMAX / TAU * VMAX / 4


def _trace(
    hz: float = 100, duration: float = 6.0, noise: float = 0.0, lead: float = 0.0, seed: int = 1
):
    t = np.arange(0, duration, 1 / hz)
    v = np.array([TRUTH.velocity_m_s(max(float(x) - lead, 0.0)) for x in t])
    if noise:
        v = v + np.random.default_rng(seed).normal(0, noise, len(v))
    return t, v


def test_noiseless_velocity_fit_recovers_parameters_and_fv_quantities() -> None:
    t, v = _trace()
    fit = sprint.fit_velocity_trace(t, v, FvSpec(mass_kg=MASS), StartSpec(), source="radar")
    assert fit.identifiable and fit.refusals == []
    assert fit.vmax_m_s == pytest.approx(VMAX, rel=1e-6)
    assert fit.tau_s == pytest.approx(TAU, rel=1e-6)
    assert fit.f0_n == pytest.approx(F0, rel=1e-6)
    assert fit.pmax_w == pytest.approx(PMAX, rel=1e-6)
    assert fit.fv_slope_n_per_m_s == pytest.approx(-F0 / VMAX, rel=1e-6)
    assert fit.rmse < 1e-6 and fit.r_squared == pytest.approx(1.0)
    assert any("zero air drag" in a for a in fit.assumptions)


def test_noisy_fit_is_within_reported_uncertainty_and_reproducible() -> None:
    t, v = _trace(noise=0.05)
    spec = FvSpec(mass_kg=MASS)
    fit = sprint.fit_velocity_trace(t, v, spec, StartSpec(), source="radar")
    again = sprint.fit_velocity_trace(t, v, spec, StartSpec(), source="radar")
    assert fit == again  # deterministic
    assert abs(fit.vmax_m_s - VMAX) < 3 * fit.parameter_se["vmax"]  # type: ignore[operator]
    assert abs(fit.tau_s - TAU) < 3 * fit.parameter_se["tau"]  # type: ignore[operator]
    assert fit.rmse == pytest.approx(0.05, rel=0.1)
    assert fit.correlation_vmax_tau is not None and fit.correlation_vmax_tau > 0.5


def test_short_window_is_refused_not_stabilised() -> None:
    t, v = _trace(duration=1.0)  # 0.83 time constants
    fit = sprint.fit_velocity_trace(t, v, FvSpec(mass_kg=MASS), StartSpec(), source="radar")
    assert not fit.identifiable and fit.vmax_m_s is None and fit.f0_n is None
    assert any("time constants" in r for r in fit.refusals)
    explicit = sprint.fit_velocity_trace(
        *_trace(), FvSpec(mass_kg=MASS, window_s=(0.0, 1.5)), StartSpec(), source="radar"
    )
    assert not explicit.identifiable


def test_start_definitions_are_reported_and_change_the_answer() -> None:
    t, v = _trace(lead=0.5)  # standing for 0.5 s before the effort
    spec = FvSpec(mass_kg=MASS)
    wrong = sprint.fit_velocity_trace(t, v, spec, StartSpec(kind="trace_origin"), source="radar")
    fitted = sprint.fit_velocity_trace(
        t, v, FvSpec(mass_kg=MASS, fit_t0=True), StartSpec(), source="radar"
    )
    assert fitted.identifiable
    assert fitted.t0_s == pytest.approx(0.5, abs=1e-3)
    assert fitted.vmax_m_s == pytest.approx(VMAX, rel=1e-3) and fitted.tau_s == pytest.approx(
        TAU, rel=1e-3
    )
    assert wrong.rmse is not None and wrong.rmse > 0.1  # a wrong start leaves structured residuals
    trigger = sprint.fit_velocity_trace(
        t, v, spec, StartSpec(kind="external_trigger", trigger_s=0.5), source="radar"
    )
    assert trigger.vmax_m_s == pytest.approx(VMAX, rel=1e-6)
    sens = sprint.fv_start_sensitivity(
        t,
        v,
        spec,
        {"origin": StartSpec(), "trigger": StartSpec(kind="external_trigger", trigger_s=0.5)},
        source="radar",
    )
    assert sens["trigger"]["f0_n"] == pytest.approx(F0, rel=1e-6)  # type: ignore[index]
    with pytest.raises(ValueError, match="trigger time"):
        sprint.start_time(t, v, StartSpec(kind="external_trigger"))
    with pytest.raises(ValueError, match="never reaches"):
        sprint.start_time(t, v, StartSpec(kind="velocity_threshold", threshold_m_s=50))


def test_milestone_times_match_exact_position_inverse() -> None:
    t, v = _trace(hz=500)
    got = sprint.milestones(t, v, [10, 20, 30], StartSpec())
    for d in (10, 20, 30):
        assert got[f"{d}_m"] == pytest.approx(TRUTH.gate_time_s(d), abs=2e-3)
    assert sprint.milestones(t, v, [1000.0], StartSpec())["1000_m"] is None


def test_distance_fit_from_gate_splits_and_refusal_with_too_few_points() -> None:
    distances = [5.0, 10.0, 20.0, 30.0, 40.0]
    times = [TRUTH.gate_time_s(d) for d in distances]
    fit = sprint.fit_split_times(distances, times, FvSpec(mass_kg=MASS, initial_tau_s=1.0))
    assert fit.vmax_m_s == pytest.approx(VMAX, rel=1e-4) and fit.tau_s == pytest.approx(
        TAU, rel=1e-4
    )
    assert fit.f0_n == pytest.approx(F0, rel=1e-3)
    over = sprint.fit_split_times(distances[:3], times[:3], FvSpec(mass_kg=MASS, fit_t0=True))
    assert not over.identifiable and any("cannot identify" in r for r in over.refusals)
    with pytest.raises(ValueError, match="at least three"):
        sprint.fit_split_times(distances[:2], times[:2], FvSpec(mass_kg=MASS))
    with pytest.raises(ValueError, match="increase"):
        sprint.fit_split_times(distances[:3], [1.0, 1.0, 2.0], FvSpec(mass_kg=MASS))


def test_gates_never_yield_instantaneous_velocity_and_sources_stay_labelled() -> None:
    with pytest.raises(ValueError, match="segment averages"):
        sprint.instantaneous_velocity("gates")
    t, v = _trace()
    with pytest.raises(ValueError, match="segment averages"):
        sprint.fit_velocity_trace(t, v, FvSpec(mass_kg=MASS), StartSpec(), source="gates")
    labels = {s: sprint.source_estimand(s)["estimand"] for s in sprint.SOURCE_ESTIMANDS}
    assert len(set(labels.values())) == len(labels)  # every source has its own definition
    assert "Doppler" in str(labels["radar"]) and "range-rate" in str(labels["laser"])
    assert "Doppler" in str(labels["gnss_doppler"]) and "differentiated" in str(
        labels["gnss_position"]
    )
    with pytest.raises(ValueError, match="unknown sprint source"):
        sprint.source_estimand("vendor")
    assert any("not instantaneous velocities" in x for x in sprint.LIMITATIONS)


def test_spec_validation() -> None:
    with pytest.raises(ValidationError):
        FvSpec(mass_kg=-1)
    with pytest.raises(ValidationError):
        FvSpec(mass_kg=80, drag="quadratic")  # type: ignore[arg-type]


def _cod_profile(hz: float = 500):
    """Closed-form COD profile: exponential acceleration to 10 m, linear braking to 1 m/s in 1 s,
    linear re-acceleration to 5 m/s in 1.2 s, then constant 5 m/s."""
    t_a = TRUTH.gate_time_s(10.0)
    v_a = TRUTH.velocity_m_s(t_a)
    d_turn = 10.0 + (v_a + 1.0) / 2 * 1.0
    d_after = d_turn + (1.0 + 5.0) / 2 * 1.2
    t_turn = t_a + 1.0
    t_regain_end = t_turn + 1.2
    total_distance = 25.0
    t_end = t_regain_end + (total_distance - d_after) / 5.0
    t = np.arange(0, t_end + 0.5, 1 / hz)
    v = np.where(
        t <= t_a,
        [TRUTH.velocity_m_s(float(x)) for x in t],
        np.where(
            t <= t_turn,
            v_a + (1.0 - v_a) * (t - t_a) / 1.0,
            np.where(t <= t_regain_end, 1.0 + 4.0 * (t - t_turn) / 1.2, 5.0),
        ),
    )
    return t, v, dict(t_a=t_a, v_a=v_a, d_turn=d_turn, t_turn=t_turn, t_end=t_end)


def test_cod_trace_metrics_match_piecewise_closed_form() -> None:
    t, v, c = _cod_profile()
    spec = CodSpec(
        declared_path="10 m sprint, 1.0 s brake, 180 deg turn, 15 m exit",
        path_length_m=25.0,
        turn_distance_m=c["d_turn"],
        turn_angle_deg=180,
        source="radar",
        straight_reference_time_s=TRUTH.gate_time_s(25.0),
    )
    result = cod.assess_trace(t, v, spec, StartSpec())
    assert result["minimum_speed_m_s"] == pytest.approx(1.0, abs=0.02)
    assert result["turn_time_s"] == pytest.approx(c["t_turn"], abs=3e-3)
    assert result["pre_turn_peak_speed_m_s"] == pytest.approx(c["v_a"], abs=0.02)
    assert result["deceleration_time_s"] == pytest.approx(1.0, abs=3e-3)
    assert result["reacceleration_time_s"] == pytest.approx((4.5 - 1.0) / (4.0 / 1.2), abs=3e-3)
    assert result["post_turn_peak_speed_m_s"] == pytest.approx(5.0)
    assert result["speed_loss_fraction"] == pytest.approx(1 - 1.0 / c["v_a"], abs=2e-3)
    assert result["total_time_s"] == pytest.approx(c["t_end"], abs=3e-3)
    assert result["cod_deficit_s"] == pytest.approx(c["t_end"] - TRUTH.gate_time_s(25.0), abs=3e-3)
    assert result["entry_speed_m_s"] > result["minimum_speed_m_s"]


def test_cod_requires_declared_path_and_reference_for_deficit() -> None:
    t, v, c = _cod_profile()
    base = dict(
        declared_path="10 m sprint, 180 deg turn",
        path_length_m=25.0,
        turn_distance_m=c["d_turn"],
        turn_angle_deg=180,
        source="radar",
    )
    no_ref = cod.assess_trace(t, v, CodSpec(**base), StartSpec())  # type: ignore[arg-type]
    assert (
        no_ref["cod_deficit_s"] is None
        and "no declared straight-sprint" in no_ref["refusals"]["cod_deficit"]
    )  # type: ignore[index]
    with pytest.raises(ValidationError):
        CodSpec(**{**base, "declared_path": ""})  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="entry distance"):
        CodSpec(**{**base, "entry_distance_m": 20.0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ends before"):
        cod.assess_trace(t, v, CodSpec(**{**base, "path_length_m": 500.0}), StartSpec())  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="source"):  # gate data cannot feed a speed trace
        CodSpec(**{**base, "source": "gates"})  # type: ignore[arg-type]


def test_cod_gates_give_segment_times_but_no_speeds() -> None:
    spec = CodSpec(
        declared_path="5-0-5 shuttle",
        path_length_m=15.0,
        turn_distance_m=10.0,
        turn_angle_deg=180,
        source="radar",
        straight_reference_time_s=2.6,
    )
    result = cod.assess_gates([0.0, 5.0, 10.0, 15.0], [0.0, 1.1, 2.9, 4.2], spec)
    assert result["total_time_s"] == pytest.approx(4.2)
    assert [s["time_s"] for s in result["segments"]] == pytest.approx([1.1, 1.8, 1.3])  # type: ignore[attr-defined]
    assert result["cod_deficit_s"] == pytest.approx(4.2 - 2.6)
    assert result["entry_speed_m_s"] is None and "not manufactured" in result["refusals"]["speeds"]  # type: ignore[index]
    with pytest.raises(ValueError, match="last gate"):
        cod.assess_gates([0.0, 5.0, 12.0], [0.0, 1.0, 2.0], spec)
    with pytest.raises(ValueError, match="increase"):
        cod.assess_gates([0.0, 5.0, 15.0], [0.0, 2.0, 1.0], spec)


def test_native_radar_to_fv_pipeline_and_cli(tmp_path: Path) -> None:
    case = export_case("radar_laser.radar", hz=100)
    options = ProcessingOptions(
        channel_stage=ChannelStage.NATIVE, correction_method="declared_fixed_angle"
    )
    result = process(case.observation, options)
    t = np.asarray(result.time_from_start_s)
    v = np.asarray(result.velocity_m_s)
    fit = sprint.fit_velocity_trace(t, np.abs(v), FvSpec(mass_kg=MASS), StartSpec(), source="radar")
    assert fit.identifiable
    assert fit.vmax_m_s == pytest.approx(VMAX, rel=0.02) and fit.tau_s == pytest.approx(
        TAU, rel=0.03
    )
    assert fit.f0_n == pytest.approx(F0, rel=0.05)

    stem = tmp_path / "radar"
    case.write(stem)
    engine_out = tmp_path / "velocity.json"
    code = main(
        [
            "radar-laser",
            "process",
            str(stem / "observation"),
            "--stage",
            "native",
            "--correction",
            "declared_fixed_angle",
            "--out",
            str(engine_out),
        ]
    )
    assert code == 0
    config = tmp_path / "fv.json"
    config.write_text(json.dumps({"source": "radar", "fv": {"mass_kg": MASS}}))
    out = tmp_path / "fv_report.json"
    assert (
        main(["assess", "sprint-fv", str(engine_out), "--config", str(config), "--out", str(out)])
        == 0
    )
    report = json.loads(out.read_text())
    assert report["fit"]["identifiable"] is True
    assert report["fit"]["f0_n"] == pytest.approx(F0, rel=0.05)
    assert "Doppler" in report["source"]["estimand"]
    assert (
        main(
            ["assess", "sprint-fv", str(engine_out), "--config", str(config), "--out", str(config)]
        )
        == 2
    )
    gates = tmp_path / "gates.json"
    gates.write_text(json.dumps({"source": "gates", "fv": {"mass_kg": MASS}}))
    assert (
        main(["assess", "sprint-fv", str(engine_out), "--config", str(gates), "--out", str(out)])
        == 2
    )
    cod_cfg = tmp_path / "cod.json"
    cod_cfg.write_text(
        json.dumps(
            {
                "cod": {
                    "declared_path": "straight 30 m",
                    "path_length_m": 30,
                    "turn_distance_m": 20,
                    "turn_angle_deg": 45,
                    "source": "radar",
                }
            }
        )
    )
    assert main(
        ["assess", "cod", str(engine_out), "--config", str(cod_cfg), "--out", str(out)]
    ) in {0, 2}


def test_modules_never_import_reference_substrate() -> None:
    for module in (sprint, cod):
        tree = ast.parse(Path(module.__file__ or "").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("dynamisbench_soccer.reference")
    assert math.isfinite(F0)


@pytest.mark.parametrize(
    "start",
    [
        StartSpec(kind="external_trigger", trigger_s=0.25),
        StartSpec(kind="velocity_threshold", threshold_m_s=0.5),
    ],
)
def test_cod_integrates_and_inverts_fractional_start(start) -> None:
    # v=2t implies distance t²-0.25², independently of the coarse sample grid.
    t = np.arange(0, 4, 0.5)
    spec = CodSpec(
        declared_path="graded turn",
        path_length_m=6,
        turn_distance_m=3,
        entry_distance_m=1,
        turn_window_m=6,
        turn_angle_deg=90,
        source="radar",
    )
    out = cod.assess_trace(t, 2 * t, spec, start)
    assert out["total_time_s"] == pytest.approx(math.sqrt(6.0625) - 0.25)
    assert out["milestones_s"] == pytest.approx(
        {"3_m": 1.75 - 0.25, "6_m": math.sqrt(6.0625) - 0.25}
    )
    assert out["entry_speed_m_s"] == pytest.approx(2 * math.sqrt(2.0625))
    assert out["post_turn_peak_speed_m_s"] == pytest.approx(2 * math.sqrt(6.0625))


@pytest.mark.parametrize(
    ("positions", "times", "reason"),
    [
        ([5, 10, 15], [0, 1, 2], "position zero"),
        ([0, 10, 15], [1, 2, 3], "elapsed time zero"),
        ([0, 12, 8, 15], [0, 1, 2, 3], "positions must increase"),
        ([0, 8, 8, 15], [0, 1, 2, 3], "positions must increase"),
        ([0, float("nan"), 15], [0, 1, 2], "finite"),
    ],
)
def test_cod_refuses_invalid_full_course_gates(positions, times, reason) -> None:
    spec = CodSpec(
        declared_path="out and back",
        path_length_m=15,
        turn_distance_m=8,
        turn_angle_deg=180,
        source="radar",
    )
    with pytest.raises(ValueError, match=reason):
        cod.assess_gates(positions, times, spec)


@pytest.mark.parametrize("trigger", [0.0625, 0.125, 0.1875])
def test_cod_constant_speed_does_not_add_remaining_sample_interval(trigger) -> None:
    time = np.arange(0.0, 3.0, 0.25)
    spec = CodSpec(
        declared_path="constant-speed control",
        path_length_m=5,
        turn_distance_m=3,
        entry_distance_m=1,
        turn_window_m=2,
        turn_angle_deg=90,
        source="radar",
        straight_reference_time_s=2.25,
    )
    result = cod.assess_trace(
        time, np.full_like(time, 2), spec, StartSpec(kind="external_trigger", trigger_s=trigger)
    )
    assert result["total_time_s"] == pytest.approx(2.5)
    assert result["milestones_s"] == pytest.approx({"3_m": 1.5, "5_m": 2.5})
    assert result["cod_deficit_s"] == pytest.approx(0.25)
