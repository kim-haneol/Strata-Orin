#!/usr/bin/env bash
# Root: decode-shaped hyper-connection timing (gr_parity STRATA_BENCH_GR) and Nsight Compute of its kernels at a
# locked GPU clock floor, restored on exit.  Usage: bench-strata-gyro-gr.sh OUT_DIR BUILD_DIR...
set -euo pipefail
B=${GYRO_RUN:?set GYRO_RUN to the prepared runtime directory}
command -v astop >/dev/null
astop agent-summary
out=${1:?output directory}
shift
D=${GYRO_GPU_DEVFREQ:-/sys/class/devfreq/17000000.gpu}
mkdir -p "$out"
before=$(cat "$D/min_freq")
trap 'echo "$before" > "$D/min_freq"; echo "restored_min_freq=$(cat "$D/min_freq")"; astop agent-summary || true' EXIT
cat "$D/max_freq" > "$D/min_freq"
for rep in 1 2; do
    for b in "$@"; do
        for T in 1 2 3 4; do
            STRATA_BENCH_GR=$T:300 "$B/$b/gr_parity" --selftest 2>&1 | grep "bench gr" | sed -E "s/^ +//; s|^|rep$rep $b |"
        done
    done
done | tee "$out/bench.txt"
if [[ -n ${NCU:-} ]]; then
    for b in "$@"; do
        for T in 1 3; do
            # the selftest's own multi calls run first; the bench's run after: skip into the timed loop
            STRATA_BENCH_GR=$T:40 timeout 900 /usr/local/cuda/bin/ncu --clock-control none \
                --kernel-name regex:"gr_up_multi_kernel|gr_down_staged_kernel" --launch-skip 60 --launch-count 4 \
                --set full --import-source no -f -o "$out/ncu-$b-T$T" "$B/$b/gr_parity" --selftest > "$out/ncu-$b-T$T.log" 2>&1
            echo "ncu $b T=$T rc=$?"
        done
    done
fi
