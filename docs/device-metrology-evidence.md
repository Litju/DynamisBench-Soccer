# Device metrology evidence audit (RES-694, 2026-11-14)

Eight primary sources were matched by DOI, title and first author against Crossref.
Compact metadata receipts and exact query endpoints are retained in
`registry/verification/RES-694-crossref.json`. Crossref verifies bibliographic
identity; the claims were checked against the primary papers separately.
All claims remain unreviewed by scientific experts.

| Claims | Device and source | Scope and retained limitation |
| -- | -- | -- |
| DEV-001 | GymAware, [Wadhi 2018](https://doi.org/10.3390/sports6040177) | SJ/CMJ comparison and repeatability; tether displacement is not the force-plate estimand |
| DEV-002 | Wearable IMU, [Toft Nielsen 2019](https://doi.org/10.1016/j.asmart.2018.09.002) | Three jump methods and two criteria; placement and algorithm specific |
| DEV-003 | Conrad contact mat, [Tenelsen 2019](https://doi.org/10.3390/sports7050114) | Drop-jump contact-time agreement/error; no flight-time generalization |
| DEV-004 | Sprint gates, [Haugen 2014](https://doi.org/10.1519/JSC.0000000000000415) | Single/dual-beam comparison; no universal height/start correction |
| DEV-005 | GPS/radar LPS, [Linke 2018](https://doi.org/10.1371/journal.pone.0199519) | Vicon comparison in male U19 soccer; systems and actions matter |
| DEV-006 | Smart Traction IMTP, [Montoro-Bombú 2023](https://doi.org/10.3390/s23135832) | Concurrent plate comparison; calibration and rig compliance do not generalize |
| DEV-007 | MuscleLAB Laser Speed, [Duthie 2025](https://doi.org/10.1007/s12283-025-00509-4) | Vicon speed comparison during straight fast runs; processed output only |
| DEV-008 | Five sprint systems, [Fornasier-Santos 2022](https://doi.org/10.3390/s22228610) | Agreement/repeatability; maximal velocity and early acceleration differ |

Toft Nielsen appeared online in 2018; Crossref assigns the volume publication to
2019. The registry uses Crossref's publication year and retains the DOI unchanged.

Every one of the 15 subfamilies links I/J/L/O to registered claims and a specific
`metrology.<subfamily>` gap. Force-platform, Nordic and hip protocol references
already in the registry are retained, without recasting them as generic hardware
calibration evidence. Existing `iso_ham` remains an open gap.

No registered source provides a universal single/dual cable correction, generic
rig-compliance uncertainty, all-device latency calibration, proprietary IMU load
algorithm, or firmware-independent raw-trace agreement. These remain gaps rather
than invented algorithms, references or calibration values. Reported correlations
and repeatability do not establish interchangeability or absolute accuracy.

Searches covered LPT/encoder validity and cable geometry; IMTP load-cell/plate
comparisons; contact mats; single/dual gate beams; laser/radar speed and sampling;
GNSS/LPS position, Doppler and acceleration; and IMU jump placement/methods.
Manufacturer validation pages and review citations were discovery aids only.
Unverified candidates (including additional dual-cable/calibration papers) were
excluded. This is a bounded registry extension, not an exhaustive systematic review.

No qualification state or comparison verdict is promoted by this audit. Published
studies are candidate empirical evidence for future engines and uncertainty models;
M3 implementation verification and M8 independent scientific qualification remain open.
