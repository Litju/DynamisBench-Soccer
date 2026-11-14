# Dataset manifests

One JSON manifest per dataset version (`<dataset_id>.json`, schema version 1). See
`docs/data-policy.md` and `src/dynamisbench_soccer/data/manifest.py`. Usage
decisions come from `docs/data-audit.md`.

Manifests contain hashes and metadata only. The data itself is never committed.
