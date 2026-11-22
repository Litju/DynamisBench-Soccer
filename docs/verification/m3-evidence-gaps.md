# M3 computational evidence gaps (RES-675)

Baseline `81fd3255eb4dde7bf9baea851682722d6f91ac94`. This inventory records the
147 unverified A–M/O cells at that commit. Owners and ratings come from
`qualification/ledger.toml`. Prior verdicts and requested corrections come from
`qualification/receipt_audit.json`. For cells with no draft, the final column
specifies the independent evidence required by the approved Policy A contract.

| Receipt ID | Existing owner issue | Prior verdict | Engineering cause | Evidence required |
| --- | --- | --- | --- | --- |
| RCP-force_platform-dual-A | RES-670 | PARTIAL | Subfamily isolation and admission | Add adapter admission test and non-finite wrench refusal for dual. |
| RCP-force_platform-dual-B | RES-670 | PARTIAL | Subfamily isolation and calibration | Cite/add dual invalid-calibration and contradictory-geometry refusal tests. |
| RCP-force_platform-dual-E | RES-670 | PARTIAL | Method variants and numerical bounds | Record 0.15 m/s / 0.1 m; assert sensitivity output. |
| RCP-force_platform-dual-F | RES-670 | PARTIAL | Numerical reconstruction | Re-issue with 0.07 m tolerance. |
| RCP-force_platform-dual-G | RES-671 | PARTIAL | Subfamily isolation and protocol events | Run dual export through process()->JumpSignal->assess for CMJ events and bilateral split. |
| RCP-force_platform-dual-H | RES-671 | PARTIAL | Method definitions and registry bindings | Add dual-fixture definition test incl. bilateral asymmetry, bound to registry claims. |
| RCP-force_platform-dual-I | RES-670 | UNSUPPORTED | Numerical uncertainty | Dual-fixture test: per-plate calibration bounds propagate to envelope matching closed form. |
| RCP-force_platform-dual-J | RES-669 | PARTIAL | Independent reference and numerical bounds | Record 0.07 m tolerance. |
| RCP-force_platform-dual-K | RES-669 | PARTIAL | QC verdicts and fault coverage | Parametrize remaining continuous/protocol fault classes on dual export with expected verdicts. |
| RCP-force_platform-dual-L | RES-703 | NO DRAFT | Cross-device agreement | Execute a dual-plate engine in a paired flight-time or impulse comparison, check analytical bias and classify under controlled timing and plate assignment. |
| RCP-force_platform-dual-O | RES-670 | NO DRAFT | Limitations | Source and emit dual-plate limitations for non-coplanar geometry, cross-talk and assignment, and assert them on dual results. |
| RCP-force_platform-single-A | RES-670 | PARTIAL | Subfamily isolation and admission | Parametrize the bad-timebase test over single; add a vendor-adapter admission test and a non-finite refusal test for a single plate. |
| RCP-force_platform-single-B | RES-670 | PARTIAL | Subfamily isolation and calibration | Cite test_invalid_calibration_rejects and metadata-frame rejection in B; add single-plate zero-window test. |
| RCP-force_platform-single-C | RES-693 | PARTIAL | Numerical timing and synchronisation | Add single-plate test: declared latency/clock offset shifts time_s exactly; cite it. |
| RCP-force_platform-single-D | RES-670 | PARTIAL | QC verdicts | Inject clipping/offset/step-off into single export and assert reject vs clean pass (e.g. cite test_device_rejection_is_carried_into_the_assessment). |
| RCP-force_platform-single-E | RES-670 | PARTIAL | Method variants and numerical bounds | Record tolerance 0.15 m/s / 0.1 m; assert reported inter-variant sensitivity. |
| RCP-force_platform-single-F | RES-670 | PARTIAL | Numerical reconstruction | Re-issue with tolerance 0.07 m (velocity 0.06 m/s). |
| RCP-force_platform-single-H | RES-671 | PARTIAL | Method definitions and registry bindings | Add test mapping each jump estimand to a registry claim id and checking the formula matches it. |
| RCP-force_platform-single-I | RES-670 | PARTIAL | Numerical uncertainty | Assert envelope equals the closed form for declared bounds; make the receipt numeric. |
| RCP-force_platform-single-J | RES-669 | PARTIAL | Independent reference and numerical bounds | Record 0.07 m tolerance. |
| RCP-force_platform-single-K | RES-669 | PARTIAL | QC verdicts and fault coverage | Parametrize OFFSET/CLIPPING/DROP/STEP_OFF on single export with expected process() verdicts. |
| RCP-force_platform-single-O | RES-670 | NO DRAFT | Limitations | Source and emit single-plate limitations, including unresolved external support and model-dependent integration, and assert them on the result. |
| RCP-gnss_lps-gnss-B | RES-700 | PARTIAL | Subfamily isolation and calibration | Cite the enu_origin=None case and add contradictory geometry or antenna-offset refusals. |
| RCP-gnss_lps-gnss-D | RES-700 | PARTIAL | QC verdicts | Cite test_admit_native_times_clock_latency_and_positions[gnss] as the clean control. |
| RCP-gnss_lps-gnss-E | RES-700 | UNSUPPORTED | Method variants and numerical bounds | Parametrize the E tests over gnss and compare each variant with the GeodeticTrack truth. |
| RCP-gnss_lps-gnss-H | RES-673 | PARTIAL | Method definitions and registry bindings | Add a registry-binding assertion once RES-694 lands. |
| RCP-gnss_lps-gnss-I | RES-700 | UNSUPPORTED | Numerical uncertainty | Propagate a GNSS position σ to speed and acceleration and compare with the closed-form variance. |
| RCP-gnss_lps-gnss-M | RES-700 | UNSUPPORTED | Provenance | Round-trip through io.write/read, assert a non-None data_sha256 equal to the file hash, and assert that the digest changes on mutation. |
| RCP-gnss_lps-gnss-O | RES-700 | PARTIAL | Limitations | Bind the limitations to sourced records. |
| RCP-gnss_lps-lps-A | RES-693 | PARTIAL | Subfamily isolation and admission | Parametrize the contradictory and resampling tests over lps. |
| RCP-gnss_lps-lps-B | RES-700 | PARTIAL | Subfamily isolation and calibration | Cite test_admit_native_times_clock_latency_and_positions[lps] (anchors and positions at 4e-9). |
| RCP-gnss_lps-lps-C | RES-693 | PARTIAL | Numerical timing and synchronisation | Parametrize the sync-pulse refusal over lps. |
| RCP-gnss_lps-lps-D | RES-700 | PARTIAL | QC verdicts | Cite a clean LPS admission control. |
| RCP-gnss_lps-lps-E | RES-700 | PARTIAL | Method variants and numerical bounds | Compare five_point and binomial5 against the PlanarTrajectory truth with stated bounds. |
| RCP-gnss_lps-lps-G | RES-673 | UNSUPPORTED | Method definitions and protocol events | Add an LPS horizontal_positional_speed threshold test with analytically known run times and dwell refusals. |
| RCP-gnss_lps-lps-H | RES-673 | PARTIAL | Method definitions and registry bindings | Add an LPS-specific estimand and registry-binding assertion. |
| RCP-gnss_lps-lps-I | RES-700 | UNSUPPORTED | Numerical uncertainty | Compare the propagated noise-to-derivative σ with the closed-form finite-difference variance. |
| RCP-gnss_lps-lps-J | RES-669 | PARTIAL | Subfamily isolation and independent reference | Replace the geodetic citations with an LPS trilateration or planar closed-form test plus an import check. |
| RCP-gnss_lps-lps-L | RES-703 | PARTIAL | Cross-device agreement | Assert radar_vs_lps_speed sensitivities against the latency closed form. |
| RCP-gnss_lps-lps-M | RES-700 | UNSUPPORTED | Provenance | Use the same fix as gnss-M, parametrized for lps. |
| RCP-gnss_lps-lps-O | RES-700 | PARTIAL | Limitations | Bind the limitations to sourced records. |
| RCP-imu-inertial-A | RES-693 | PARTIAL | Subfamily isolation and admission | Add stage, unit-label and non-finite or non-monotonic input refusals for imu.inertial. |
| RCP-imu-inertial-C | RES-693 | NO DRAFT | Numerical timing and synchronisation | Apply declared IMU timestamps, synchronisation and latency to a reference trajectory, propagate timing uncertainty, and refuse unresolved or inconsistent alignment. |
| RCP-imu-inertial-G | RES-671 | PARTIAL | Method definitions and protocol events | Record the event tolerance as 2/hz = 0.004 s. |
| RCP-imu-inertial-H | RES-671 | PARTIAL | Method definitions and registry bindings | Add a registry-binding check once the claims exist. |
| RCP-imu-inertial-K | RES-669 | PARTIAL | QC verdicts and fault coverage | Cite test_static_gyro_bias_and_slip_are_flagged and add a misalignment verdict. |
| RCP-imu-inertial-L | RES-703 | PARTIAL | Cross-device agreement | Record tolerance 0.006 s. |
| RCP-imu-inertial-M | RES-701 | NO DRAFT | Provenance | Verify source hash, placement and calibration declarations in IMU output with exact replay and input-mutation detection. |
| RCP-imu-inertial-O | RES-701 | PARTIAL | Limitations | Bind the limitations to sourced records. |
| RCP-load_cell_isometric-isometric_rig-B | RES-696 | PARTIAL | Subfamily isolation and calibration | Cite the passthrough test; add rig-fixture lever-arm/torque test. |
| RCP-load_cell_isometric-isometric_rig-C | RES-693 | PARTIAL | Numerical timing and synchronisation | Rig-fixture timing refusal and alignment test. |
| RCP-load_cell_isometric-isometric_rig-D | RES-696 | PARTIAL | QC verdicts | Inject offset/clipping into isometric_rig export and assert verdicts. |
| RCP-load_cell_isometric-isometric_rig-E | RES-696 | PARTIAL | Method variants and numerical bounds | Rig-fixture variant test; record 0.03 bounds. |
| RCP-load_cell_isometric-isometric_rig-F | RES-696 | PARTIAL | Subfamily isolation and reconstruction | Run analytical metric recovery on isometric_rig export. |
| RCP-load_cell_isometric-isometric_rig-G | RES-672 | PARTIAL | Subfamily isolation and protocol events | Parametrize _bundle over isometric_rig export. |
| RCP-load_cell_isometric-isometric_rig-H | RES-672 | PARTIAL | Method definitions and registry bindings | Run the isometric_rig export through assess/process against the IsometricRamp closed forms, and tighten the variant RFD bound. |
| RCP-load_cell_isometric-isometric_rig-I | RES-696 | PARTIAL | Numerical uncertainty | Repeat the gain and onset sensitivity checks on export_case('load_cell_isometric.isometric_rig'). |
| RCP-load_cell_isometric-isometric_rig-J | RES-669 | PARTIAL | Subfamily isolation and independent reference | Add a parametrisation that runs the isometric_rig export through process and compares it to the closed form. |
| RCP-load_cell_isometric-isometric_rig-K | RES-669 | UNSUPPORTED | QC verdicts and fault coverage | Inject offset, clipping and units faults into the isometric_rig force channel and assert the LC-* verdicts. |
| RCP-load_cell_isometric-isometric_rig-L | RES-703 | NO DRAFT | Cross-device agreement | Execute the isometric-rig engine in a paired force comparison, assert exact ideal agreement and classify geometry or onset mismatch. |
| RCP-load_cell_isometric-load_cell-C | RES-693 | PARTIAL | Numerical timing and synchronisation | Bundle with non-reference clock alignment: assert time shift and onset_resolution_s closed form. |
| RCP-load_cell_isometric-load_cell-E | RES-696 | PARTIAL | Method variants and numerical bounds | Record tolerance rel 0.03 / 0.03 s. |
| RCP-load_cell_isometric-load_cell-H | RES-672 | PARTIAL | Method definitions and registry bindings | Add registry-claim binding test for isometric metrics. |
| RCP-load_cell_isometric-load_cell-I | RES-696 | PARTIAL | Numerical uncertainty | Assert declared calibration/sync uncertainty propagate to reported values per closed form. |
| RCP-load_cell_isometric-load_cell-K | RES-669 | PARTIAL | QC verdicts and fault coverage | Parametrize all ContinuousKind on load_cell export with expected process() verdicts. |
| RCP-lpt_encoder-encoder-A | RES-693 | UNSUPPORTED | Subfamily isolation and admission | Encoder-fixture refusal tests (duplicate timestamps, bad count calibration units, analytical truth input). |
| RCP-lpt_encoder-encoder-C | RES-693 | NO DRAFT | Numerical timing and synchronisation | Apply a declared encoder clock/latency transform and compare event timestamps with the affine oracle. Refuse a contradictory clock or non-monotonic corrected timestamps. |
| RCP-lpt_encoder-encoder-D | RES-695 | PARTIAL | QC verdicts | Encoder-fixture dropout/duplicate faults and clean-control assertion. |
| RCP-lpt_encoder-encoder-E | RES-695 | NO DRAFT | Method variants and numerical bounds | Compare every supported count-unwrapping and rate method with independent signed count histories. Bound the sensitivity to count quantisation and timestamp resolution. |
| RCP-lpt_encoder-encoder-F | RES-695 | NO DRAFT | Numerical reconstruction | Recover displacement and rate from a signed, wrapped count history with known counter scale, direction and timestamp spacing. Derive the count and timing bounds. |
| RCP-lpt_encoder-encoder-G | RES-671 | NO DRAFT | Method definitions and protocol events | Feed encoder-reconstructed displacement/rate into the applicable jump event path and assert analytical event times and eligibility refusals. |
| RCP-lpt_encoder-encoder-H | RES-671 | NO DRAFT | Method definitions and registry bindings | Bind count-based displacement and rate estimands to sourced registry claims. Keep encoder attachment displacement distinct from centre-of-mass displacement. |
| RCP-lpt_encoder-encoder-I | RES-695 | NO DRAFT | Numerical uncertainty | Propagate declared count and clock quantisation through displacement and rate, checking the observed bounds against the analytical result. |
| RCP-lpt_encoder-encoder-J | RES-669 | NO DRAFT | Independent reference and numerical bounds | Run encoder unwrapping and rate against an independent signed count oracle. Check that the engine and reference share no reconstruction code. |
| RCP-lpt_encoder-encoder-K | RES-669 | PARTIAL | QC verdicts and fault coverage | Assert acquire() outcome for each EventKind on encoder fixture. |
| RCP-lpt_encoder-encoder-L | RES-703 | NO DRAFT | Cross-device agreement | Run a paired analytical encoder comparison using matched attachment kinematics, and quantify and classify disagreement with a non-matched estimand. |
| RCP-lpt_encoder-encoder-M | RES-695 | NO DRAFT | Provenance | Round-trip encoder observations through verified I/O, trace source digest and method declarations into the report, and verify exact replay and mutation detection. |
| RCP-lpt_encoder-encoder-O | RES-695 | NO DRAFT | Limitations | Source and emit encoder limitations covering count quantisation, wrap ambiguity and attachment-versus-centre-of-mass displacement. |
| RCP-lpt_encoder-lpt-A | RES-693 | PARTIAL | Subfamily isolation and admission | Add single-cable unit/calibration-unit refusal and non-finite length refusal tests. |
| RCP-lpt_encoder-lpt-B | RES-695 | PARTIAL | Subfamily isolation and calibration | LPT-fixture test applying/refusing a cable Calibration record and contradictory attachment. |
| RCP-lpt_encoder-lpt-D | RES-695 | PARTIAL | QC verdicts | Assert faults=={} and geometry_method unchanged on clean export in the integrity test. |
| RCP-lpt_encoder-lpt-E | RES-695 | PARTIAL | Method variants and numerical bounds | Parametrize methods on single-cable LPT cable_length fixture vs closed form. |
| RCP-lpt_encoder-lpt-G | RES-671 | PARTIAL | Method definitions and protocol events | Assert LPT-derived take-off/apex times vs motion.takeoff_s() closed form. |
| RCP-lpt_encoder-lpt-H | RES-671 | PARTIAL | Method definitions and registry bindings | Add registry-claim binding test for LPT estimands. |
| RCP-lpt_encoder-lpt-I | RES-695 | UNSUPPORTED | Numerical uncertainty | LPT fixture; compare cable-angle/timebase perturbation deltas to closed-form cos/scale derivatives. |
| RCP-lpt_encoder-lpt-K | RES-669 | PARTIAL | QC verdicts and fault coverage | Seeded inject() fixtures of LPT fault classes with expected acquire() fault maps. |
| RCP-lpt_encoder-lpt-L | RES-703 | PARTIAL | Cross-device agreement | Run LPT engine in at least one study and assert its classification. |
| RCP-lpt_encoder-lpt-O | RES-695 | PARTIAL | Limitations | Assert a declared LPT LIMITATIONS set is emitted on cable-only, fixed-line and dual results. |
| RCP-nordic_hip-hip-A | RES-697 | PARTIAL | Subfamily isolation and admission | Add hip-data refusal tests: left/right declared on the hip export, a missing 'force' channel, and force_basis on hip. |
| RCP-nordic_hip-hip-B | RES-697 | UNSUPPORTED | Subfamily isolation and calibration | Cite test_hip_squeeze_metrics_and_geometry_gate and add hip torque refusal and calibration application tests. |
| RCP-nordic_hip-hip-C | RES-693 | NO DRAFT | Numerical timing and synchronisation | Compare explicit hip timestamps and declared clock alignment against the affine oracle, with inconsistent clock/time refusal and reported timing uncertainty. |
| RCP-nordic_hip-hip-D | RES-697 | UNSUPPORTED | QC verdicts | Inject faults into the hip export and assert the QC verdicts, with a clean hip control. |
| RCP-nordic_hip-hip-E | RES-697 | NO DRAFT | Method variants and numerical bounds | Verify each applicable filter, onset and repetition policy on known hip force ramps and holds. Bound the filter and repetition sensitivities and demonstrate inapplicable methods explicitly. |
| RCP-nordic_hip-hip-F | RES-697 | PARTIAL | Numerical reconstruction | Cite IsometricRamp as the reference, set tolerance 1e-5, and add hip impulse and RFD closed-form checks. |
| RCP-nordic_hip-hip-G | RES-672 | PARTIAL | Method definitions and protocol events | Assert repetition onset and peak times against the stacked IsometricRamp timings, and add a refusal for too-close or absent repetitions. |
| RCP-nordic_hip-hip-H | RES-672 | UNSUPPORTED | Method definitions and registry bindings | Cite or add hip tests that execute max-mean window definitions and unilateral estimand naming on the hip export. |
| RCP-nordic_hip-hip-I | RES-697 | UNSUPPORTED | Numerical uncertainty | Run the gain and lever-uncertainty propagation on the hip export against 2500*(1±e) and 1000*σ/0.4. |
| RCP-nordic_hip-hip-J | RES-669 | UNSUPPORTED | Subfamily isolation and independent reference | Cite IsometricRamp and test_hip_squeeze_metrics_and_geometry_gate. |
| RCP-nordic_hip-hip-K | RES-669 | UNSUPPORTED | QC verdicts and fault coverage | Add hip fault fixtures with expected engine verdicts. |
| RCP-nordic_hip-hip-L | RES-703 | NO DRAFT | Cross-device agreement | Run the hip engine and a criterion computational force or torque method on the same ideal geometry, and quantify and classify a deliberate geometry or force-basis mismatch. |
| RCP-nordic_hip-hip-M | RES-697 | NO DRAFT | Provenance | Record the verified hip bundle digest and processing declarations, and prove exact replay and hash change on mutation. |
| RCP-nordic_hip-hip-O | RES-697 | NO DRAFT | Limitations | Bind emitted hip limitations for cuff force, protocol angle and lever length to sourced records, and assert them on hip results. |
| RCP-nordic_hip-nordic-A | RES-697 | PARTIAL | Subfamily isolation and admission | Add refusal tests for the wrong-device bundle, wrong channel quantity and non-deg angle unit. |
| RCP-nordic_hip-nordic-B | RES-697 | PARTIAL | Subfamily isolation and calibration | Use a voltage nordic export with a declared gain and offset, assert the force equals the closed form, and refuse contradictory or duplicate calibration records. |
| RCP-nordic_hip-nordic-C | RES-693 | NO DRAFT | Numerical timing and synchronisation | Compare both aligned limb clocks against a declared offset and drift oracle, including an unsynchronised-pair refusal and timing uncertainty output. |
| RCP-nordic_hip-nordic-D | RES-697 | PARTIAL | QC verdicts | Make saturation or clipping raise an LC-RAILS/CAPACITY flag (or declare it undetectable) and assert that the clean _case() is accepted. |
| RCP-nordic_hip-nordic-E | RES-697 | NO DRAFT | Method variants and numerical bounds | Verify the supported filter, onset, repetition and break-point policies against independent eccentric and angle traces. State and check each sensitivity bound. |
| RCP-nordic_hip-nordic-I | RES-697 | PARTIAL | Numerical uncertainty | Assert both gain variants against PEAK*share*(1±e) and torque uncertainty against PEAK*share*0.01, and record rel 1e-6 as a numeric tolerance. |
| RCP-nordic_hip-nordic-K | RES-669 | PARTIAL | QC verdicts and fault coverage | Add a nordic fault matrix (clip, drop, offset, noise, side swap) with expected verdicts, and drop the force_platform.dual test. |
| RCP-nordic_hip-nordic-M | RES-697 | NO DRAFT | Provenance | Record hashes for the bilateral source bundle and declarations, verify byte-identical replay, and prove a one-sample change changes the digest. |
| RCP-nordic_hip-nordic-O | RES-697 | NO DRAFT | Limitations | Bind emitted Nordic limitations for ankle-strap force, lever geometry, and measured angle to sourced records and assert them on results. |
| RCP-radar_laser-laser-A | RES-693 | PARTIAL | Subfamily isolation and admission | Parametrize the refusal tests over radar_laser.laser. |
| RCP-radar_laser-laser-B | RES-699 | PARTIAL | Subfamily isolation and calibration | Run the geometry refusals and an applied-geometry closed-form check on laser bundles. |
| RCP-radar_laser-laser-D | RES-699 | PARTIAL | QC verdicts | Add a laser range-saturation fault and an explicit clean laser accept control. |
| RCP-radar_laser-laser-E | RES-699 | PARTIAL | Method variants and numerical bounds | Compare each filter and start variant with closed-form truth within stated bounds. Record tolerance 0.004 as numeric. |
| RCP-radar_laser-laser-F | RES-699 | PARTIAL | Numerical reconstruction | Record check_kind numeric, tolerance 0.004 m/s. |
| RCP-radar_laser-laser-G | RES-673 | PARTIAL | Method definitions and protocol events | Pipe radar_laser.laser process output into sprint.milestones and check against gate_time_s. |
| RCP-radar_laser-laser-H | RES-673 | PARTIAL | Method definitions and registry bindings | Execute laser-sourced fits and assert the binding to registry claims. |
| RCP-radar_laser-laser-I | RES-699 | UNSUPPORTED | Numerical uncertainty | Check the laser five-point noise amplification (sqrt(130)/12·σ·fs) against the engine's reported uncertainty. |
| RCP-radar_laser-laser-J | RES-669 | PARTIAL | Independent reference and numerical bounds | Record the laser tolerance as 0.004, or split the radar/laser J evidence. |
| RCP-radar_laser-laser-K | RES-669 | PARTIAL | QC verdicts and fault coverage | Inject each continuous fault class into laser bundles and assert the engine refusal or verdict. |
| RCP-radar_laser-laser-M | RES-699 | NO DRAFT | Provenance | Verify the laser source hash and processing declarations in native and processed results, with exact replay and input-mutation detection. |
| RCP-radar_laser-laser-O | RES-699 | PARTIAL | Limitations | Bind the limitations to sourced records and cover all stages. |
| RCP-radar_laser-radar-B | RES-699 | PARTIAL | Subfamily isolation and calibration | Cite test_radar_recovers_independent_exact_velocity_at_native_rate (cosine applied, 2e-12) or the cosine_angle assertion in the admission test. |
| RCP-radar_laser-radar-E | RES-699 | PARTIAL | Method variants and numerical bounds | Compare each radar filter and start variant with the truth velocity or time within stated bounds. |
| RCP-radar_laser-radar-I | RES-699 | PARTIAL | Numerical uncertainty | Compare the start ± sensitivity values with MonoExponentialSprint velocity at the shifted times, and use radar for the filter sensitivity. |
| RCP-radar_laser-radar-K | RES-669 | PARTIAL | QC verdicts and fault coverage | Add expected radar process/admit verdicts for the offset, drift, noise, resample and drop fixtures. |
| RCP-radar_laser-radar-L | RES-703 | NOT_AUDITED | Cross-device agreement | Independently review the radar-versus-positional-speed study on radar output. Check the analytical latency bias, the definition-mismatch classification and the unknown empirical verdict. |
| RCP-radar_laser-radar-M | RES-699 | NO DRAFT | Provenance | Verify the radar source hash and processing declarations in native and processed results, with exact replay and input-mutation detection. |
| RCP-radar_laser-radar-O | RES-699 | PARTIAL | Limitations | Assert that each limitation maps to a sourced ledger or registry entry. Parametrize over native/processed stages and correction methods. |
| RCP-timing_contact-contact-B | RES-698 | NO DRAFT | Subfamily isolation and calibration | Apply typed switch semantics and on/off declarations to known contact transitions, with missing or contradictory actuator declarations refused. |
| RCP-timing_contact-contact-C | RES-693 | PARTIAL | Numerical timing and synchronisation | Add or cite a timing refusal (missing switching definition or clock authority, unresolved latency) and the affine-clock test. |
| RCP-timing_contact-contact-G | RES-671 | PARTIAL | Method definitions and protocol events | Record a numeric tolerance (rel 1e-6). |
| RCP-timing_contact-contact-I | RES-698 | PARTIAL | Numerical uncertainty | Assert the RSI and flight interval bounds equal the closed-form propagation of the declared on and off latency errors. |
| RCP-timing_contact-contact-K | RES-669 | PARTIAL | QC verdicts and fault coverage | Run contact() on the DOUBLE_CONTACT, STEP_OFF and LATENCY fixtures and assert the expected refusal or verdict. |
| RCP-timing_contact-contact-M | RES-698 | NO DRAFT | Provenance | Verify the event-bundle hash and configuration in contact output with exact replay and input-mutation detection. |
| RCP-timing_contact-contact-O | RES-698 | NO DRAFT | Limitations | Source and emit limitations of thresholded contact, latency and the equal take-off/landing height assumption for flight-time height. |
| RCP-timing_contact-gates-B | RES-698 | NO DRAFT | Subfamily isolation and calibration | Apply typed course distances and gate geometry in the split-speed calculation. Refuse inconsistent gate placement or missing required declarations. |
| RCP-timing_contact-gates-C | RES-693 | PARTIAL | Numerical timing and synchronisation | Record a tolerance of rel 1e-6, or assert engine splits at abs 1e-11. Also cite the affine-clock round-trip splits test. |
| RCP-timing_contact-gates-E | RES-698 | PARTIAL | Method variants and numerical bounds | Record the numeric tolerance and add trigger and fixed_offset start variants against the truth time. |
| RCP-timing_contact-gates-F | RES-698 | PARTIAL | Numerical reconstruction | Set the tolerance to rel 1e-6, or tighten the engine assertions. |
| RCP-timing_contact-gates-G | RES-671 | PARTIAL | Method definitions and protocol events | Record the numeric tolerance, and feed engine splits into assess_gates. |
| RCP-timing_contact-gates-I | RES-698 | PARTIAL | Numerical uncertainty | Assert the split interval half-width equals the root-sum or linear combination of the declared errors. |
| RCP-timing_contact-gates-J | RES-669 | PARTIAL | Independent reference and numerical bounds | Correct the tolerance to the engine's asserted bound. |
| RCP-timing_contact-gates-K | RES-669 | PARTIAL | QC verdicts and fault coverage | Add a LATENCY fault case to test_adversarial_reference_faults_are_refused, or an expected detection or bias verdict for it. |
| RCP-timing_contact-gates-M | RES-698 | NO DRAFT | Provenance | Verify the event-bundle hash and configuration in gate output with exact replay and input-mutation detection. |
| RCP-timing_contact-gates-O | RES-698 | NO DRAFT | Limitations | Source and emit limitations on beam triggering, start definition and split versus instantaneous speed. |
