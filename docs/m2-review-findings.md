# M2 pathway review findings and disposition (RES-705)

The continuation requested preservation of known M2 findings. No earlier named
review report or GitHub review threads were available in this checkout/PR #3.
The following defects were independently reproduced during RES-692 review and
repaired here; this list does not assert that unavailable findings were all recovered.

| Finding | Repair and regression evidence |
| -- | -- |
| Streaming bypassed sidecar/hash validation | Both readers share metadata/hash/schema/row-count validation before samples are used |
| Full read silently coerced noncanonical storage types | Float64 storage types and exact column order are checked before conversion |
| Projection could hide invalid unselected samples | Full batches are validated before projection, in bounded memory |
| QC used nominal time despite measured timestamps | Uniform measured timing determines durations and derivatives; irregular/nonincreasing timing fails timing-dependent QC |
| Timestamp/rate diagnostics could be hidden by timing guards | S04/S05 inspect timestamps independently; S05 derives the observed rate when available |
| Zero force denominators could yield undefined QC values | Nonpositive baseline/peak force returns an explicit failure |

Read hash checking can be disabled only through the existing explicit `verify=False`
full-reader argument. Scientific reference callers must use verified reading.
The streaming API always checks the hash and validates batches; as with any stream,
a later corrupt batch can fail after earlier valid batches have been delivered.
Consumers must complete validation before promoting a whole trial to reference status.

QC supports uniformly sampled timing calculations. An irregular timestamp series
requires a separately specified/reviewed irregular-time method or explicit resampling
with provenance. The current QC path fails those calculations; it never silently
interpolates. The rules' numerical thresholds remain engineering defaults, pending
scientific review. QC shape fixtures remain physically inconsistent and cannot serve
as analytical truth. These fixes do not qualify any device, assessment or AI grader.
