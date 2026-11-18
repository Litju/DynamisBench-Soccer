#!/usr/bin/env bash
# Execute the FORCE-only adapter and existing processor in a mount allowlist.
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "usage: $0 FORCE_SOURCE_ROOT NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi
repo=$(cd "$(dirname "$0")/.." && pwd)
source_root=$(realpath -e "$1")
output=$(realpath -m "$2")
case "$output/" in
  "$source_root/"*|"$repo/"*) echo "output must be outside SOURCE and the repository" >&2; exit 2 ;;
esac
source_mounts=()
for case in trial-00000718-0000 trial-00000718-0001; do
  test ! -L "$source_root/$case" && test ! -L "$source_root/$case/streams" || exit 2
  for relative in manifest.json capture_authority.json source-verification.json streams/force.csv; do
    file="$source_root/$case/$relative"
    test -f "$file" && test ! -L "$file" && test "$(stat -c %h "$file")" -eq 1 || exit 2
    source_mounts+=(--ro-bind "$file" "/source/$case/$relative")
  done
done
mkdir "$output"
python=$(readlink -f "$repo/.venv/bin/python")
runtime=$(dirname "$(dirname "$python")")

bwrap --unshare-all --die-with-parent --clearenv --ro-bind /usr /usr \
  --symlink usr/bin /bin --symlink usr/lib /lib --symlink usr/lib64 /lib64 \
  --proc /proc --dev /dev --tmpfs /tmp \
  --ro-bind "$runtime" /runtime --ro-bind "$repo/.venv/lib/python3.12/site-packages" /packages \
  --ro-bind "$repo/src" /app "${source_mounts[@]}" --bind "$output" /out \
  --ro-bind "$repo/data/manifests/biosyngen-res718-force.json" /dataset.json \
  --ro-bind "$repo/docs/res718/method.json" /method.json \
  --ro-bind "$repo/qc/rules.toml" /qc/rules.toml \
  --setenv PYTHONPATH /app:/packages --setenv OPENBLAS_NUM_THREADS 1 --chdir /out \
  /bin/bash -euc '
    test ! -e /TRUTH
    test ! -e /source/TRUTH
    test ! -e "$1/../corpus/TRUTH"
    printf "SOURCE-only mount; host corpus path inaccessible\n" > /out/isolation.txt
    for case in trial-00000718-0000 trial-00000718-0001; do
      /runtime/bin/python3.12 -m dynamisbench_soccer.adapters.biosyngen_force "/source/$case" \
        --dataset /dataset.json --out-dir "/out/$case"
      status=0
      /runtime/bin/python3.12 -m dynamisbench_soccer.cli force-platform process "/out/$case/force" \
        --config /method.json --qc-rules /qc/rules.toml --out "/out/$case/engine.json" \
        --traces "/out/$case/traces.csv" || status=$?
      printf "%s\n" "$status" > "/out/$case/exit-code"
      test "$status" -le 1 || { cat "/out/$case/engine.json" >&2; exit "$status"; }
    done
  ' res718 "$source_root"
