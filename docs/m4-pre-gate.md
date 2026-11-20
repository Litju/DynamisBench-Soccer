# RES-737 M4 preparation

**DEVELOPMENT_ONLY_NOT_QUALIFIED**

Reproduce the four-task demonstration on unprivileged Debian/Ubuntu Linux with
system Python 3, bubblewrap and user namespaces.

```sh
uv sync --frozen
uv run python -m dynamisbench_soccer.m4.demo
uv run python -m dynamisbench_soccer.m4.demo --episodes
uv run pytest -q tests/test_m4_contracts.py tests/test_m4_runner.py tests/test_m4_demo.py
```

The CLI emits the versioned public episodes with `--episodes`. To run a supplied
Python program, use `--candidate path/to/candidate.py`. It reads one episode as
JSON on stdin and writes one marked `m4.answer.v1` JSON object on stdout. Include
the declared estimand and unit, the decision, method and reason, and the full
metric set. Refusals give an empty metric set and an evidence-specific reason.
The demo uses exact declared model assumptions and a deterministic absolute
numerical tolerance. The defended ballistic variant admits two equivalent
formulas. Graders score numerical and methodological criteria independently.

| DEVELOPMENT_ONLY_NOT_QUALIFIED task | Evidence | Oracle | Naive | Null | Random |
| --- | --- | --- | --- | --- | --- |
| Fractional COD | Analytical | Pass | Fail | Fail | Fail |
| Ballistic height | Analytical | Pass | Fail | Fail | Fail |
| Partial gate course | Synthetic | Pass (refusal) | Fail | Fail | Fail |
| Missing DJ touchdown evidence | Synthetic | Pass (refusal) | Fail | Fail | Fail |

The output carries per-episode and candidate digests, evidence class, separate
score dimensions and execution outcome. All scores and artifacts carry the
development marker. Controls exercise wrong units, swapped estimands, guessed
methods, malformed and duplicate-key JSON, unsupported vendor claims and
embedded prompt injection.

The host grader generates its independent demo keys into a private temporary
directory, checks the answer digest and public-episode binding, and scores
after the candidate exits. Bubblewrap mounts the candidate source and trusted
system Python runtime read-only. The public episode enters via stdin. The
sandbox has a separate network and PID namespace, no host environment or
project, and no writable filesystem. It limits wall time to 3 seconds,
CPU time to 1 second, address space to 128 MiB and output to 16 KiB, with
child creation disabled. Tests try to open the actual key and grader source,
import the host project, inspect the environment, connect to a live host
socket, create a process and exceed limits. This is a local prototype using the
host kernel. The reproducible demo keys are visible to repository readers
with host access. Use a private external key store for a blind evaluation.

After the real M3 gate clears, RES-676 can adapt the public/private contracts
and registry, RES-677 can extend the deterministic graders with reviewed
rubrics, and RES-678 can harden runner provenance and longitudinal episodes.
RES-679 retains responsibility for independent grader qualification and
prespecified false-pass and false-fail evaluation. This demo leaves RES-676
blocked by RES-675 and carries no M4/M5 scientific gate pass.
