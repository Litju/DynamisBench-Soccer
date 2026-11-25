# RES-704 encoder scientific reference admission

The frozen `encoder-rollover-attachment@1` manifest and five source artifacts
define a single synthetic 50 Hz, source-owned integer counter. The estimand is
signed start-to-end vertical motion of the declared bar attachment. The method
uses the native encoder acquisition and relative projection from RES-695.

| Role | Task dependency and evidence |
| --- | --- |
| A | Source-owned integer counter and explicit time columns, device and unit identity, native channel digest, and rejected malformed values |
| B | Declared identity count calibration, counts per metre, physical direction and effective angle from vertical. The existing encoder receipt supports the native engine calibration and geometry. |
| C | Exact native 50 Hz grid and endpoint elapsed time. The whole record must be ordered and complete. No interpolation. |
| D | Range and integer checks on unsigned residues, strict half-turn guarantee and per-interval physical increment limits |
| E | Native signed modular unwrapping, sign convention applied once, no differentiation or smoothing of the endpoint estimate |
| F | Independently derived count, relative cable motion, vertical displacement and signed mean velocity |
| H | Attachment-point displacement is a signed length. Cable motion, vertical motion, mean velocity and athlete COM are distinct. |
| I | Two half-count endpoint errors propagate through scale and the declared nonsingular constant projection |
| J | Independent integer lift enumeration, additional signed and wrapped controls, separate owner-private author and scientific reviewer |
| K | Exact scope refusals for absent scale and angle, failed global wrap guarantee and implausible increments. Additional integer, half-turn, timing and interpretation controls. |
| M | Canonical source digest, exact task and artifact binding, private answer and replay provenance. Existing encoder file-backed evidence supports the overlapping digest and deterministic provenance properties. |
| O | Scope is the bar attachment and conditional synthetic endpoint quantization. DEV-001 contextualizes the attachment interpretation. |

G and L are inapplicable to this single-instrument endpoint measurement. The
scoped A/C/D/E/F/H/I/J/K/O receipts pair with the existing encoder B/M
receipts. The task-specific tests verify the JSON channel hash and immutable
artifact projection alongside the existing M provenance evidence. M3 ratings
remain independent of this admission.

For every adjacent observed pair the possible increments are
`next − previous + kM` for integer `k`. When the declared allowance
`v_max × Δt × counts_per_m + 1` is strictly below `M/2`, at most one
increment can fit. The native engine chooses the centered representative and
rejects any individual increment beyond the allowance. The primary six
increments are +4 counts, so the declared −1 direction gives −24 counts.
At 400 counts/m and zero effective angle, this is −0.06 m. Native elapsed
time is 0.12 s and mean vertical velocity is −0.5 m/s. Two half-count
endpoint bounds sum to one count, or 0.0025 m after projection. The wrap
guarantee uses adjacent count error, separate from the endpoint bound.

The missing-scale fixture uses `insufficient_evidence` because the frozen
prompt assigns that reason to absent declarations. The earlier development
diagnostic used a general acquisition refusal for the same missing scale.

The faster-speed fixture gives an allowance of 161 counts, above the 128-count
limit. This fails the prespecified global guarantee even though an observed
+4 residue alone has a unique admissible lift at that allowance. The separate
half-turn witness exhibits two equally admissible directions. The last frozen
fixture keeps the same endpoints while injecting invalid interior changes.
Supplemental traces cross the counter boundary in reverse and complete an entire
turn with identical endpoint residues. They also vary the direction, angle,
scale, and clock origin.

The prespecified tolerance is 1e-9 absolute with zero relative tolerance for
the quantity, radius and numeric operands. Private replay uses canonical
10-decimal operands. Unit, estimand, method, citations, claims and operand
identities are checked exactly. All five oracle checkpoints must pass.
Prespecified controls require zero false accepts and rejects, unchanged
legacy scores, private-key isolation, deterministic replay and complete
workspace cleanup. The bound covers endpoint count quantization under the
declared scale and geometry. Clock error, slip, real attachment motion relative
to the athlete and hardware validity are outside this bound.
