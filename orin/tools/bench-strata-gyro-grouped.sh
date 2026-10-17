#!/usr/bin/env bash
# Root: decode-shaped grouped-expert timing (native_expert_parity STRATA_BENCH_GROUPED) for several builds at a
# locked GPU clock floor, alternating builds, plus parity on two Gyro-M layers. Restores the floor on exit.
# Usage: bench-strata-gyro-grouped.sh OUT_DIR BUILD_DIR...
set -euo pipefail
B=${GYRO_RUN:?set GYRO_RUN to the prepared runtime directory}
command -v astop >/dev/null
astop agent-summary
out=${1:?output directory}
shift
D=${GYRO_GPU_DEVFREQ:-/sys/class/devfreq/17000000.gpu}
G=$(grep ^GYRO_MODEL_PATH= "$B/runtime.env" | cut -d= -f2)
mkdir -p "$out"
before=$(cat "$D/min_freq")
trap 'echo "$before" > "$D/min_freq"; echo "restored_min_freq=$(cat "$D/min_freq")"; astop agent-summary || true' EXIT
cat "$D/max_freq" > "$D/min_freq"
for b in "$@"; do
    for L in 10 30; do
        "$B/$b/native_expert_parity" "$G" "$L" 2>&1 | grep -E "layer |width|decode-once|failures" | sed "s|^|$b parity L$L: |"
    done
done | tee "$out/parity.txt"
for rep in 1 2; do
    for b in "$@"; do
        for T in 1 2 4; do
            STRATA_BENCH_GROUPED=$T:300 "$B/$b/native_expert_parity" "$G" 10 2>&1 | grep "bench grouped" | sed -E "s/^ +//; s|^|rep$rep $b |"
        done
    done
done | tee "$out/bench.txt"
