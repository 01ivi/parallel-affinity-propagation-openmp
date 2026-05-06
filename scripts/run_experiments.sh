#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p results
RESULT_FILE="results/result.csv"

# 可通过环境变量覆盖默认参数，例如:
# NS="1000 3000 6000" THREADS="1 2 4 8" NEIGHBORS=40 bash scripts/run_experiments.sh
DIM="${DIM:-10}"
TRUE_K="${TRUE_K:-6}"
MAX_ITER="${MAX_ITER:-100}"
TOL="${TOL:-1e-4}"
SEED="${SEED:-42}"
NEIGHBORS="${NEIGHBORS:-30}"
DAMPING="${DAMPING:-0.75}"
PREFERENCE="${PREFERENCE:--10}"
NS="${NS:-1000 3000 6000}"
THREADS="${THREADS:-1 2 4 8}"

make

BIN="./parallel_ap"
if [[ -x "./parallel_ap.exe" ]]; then
    BIN="./parallel_ap.exe"
fi

echo "n,dim,true_k,neighbors,mode,threads,build_time,ap_time,total_time,iterations,clusters,objective,edges,preference,damping,max_delta,speedup,efficiency" > "$RESULT_FILE"

calc_div() {
    awk -v a="$1" -v b="$2" 'BEGIN {
        if (b == 0) {
            printf "0.000000";
        } else {
            printf "%.6f", a / b;
        }
    }'
}

append_result() {
    local raw="$1"
    local true_k="$2"
    local speedup="$3"
    local efficiency="$4"

    # 程序输出格式:
    # mode,n,dim,neighbors,threads,build_time,ap_time,total_time,iterations,clusters,objective,edges,preference,damping,max_delta
    IFS=',' read -r mode_out n_out dim_out neighbors_out threads_out build_out ap_out total_out iterations_out clusters_out objective_out edges_out preference_out damping_out max_delta_out <<< "$raw"
    echo "${n_out},${dim_out},${true_k},${neighbors_out},${mode_out},${threads_out},${build_out},${ap_out},${total_out},${iterations_out},${clusters_out},${objective_out},${edges_out},${preference_out},${damping_out},${max_delta_out},${speedup},${efficiency}" >> "$RESULT_FILE"
}

for n in $NS; do
    echo "Running serial baseline: n=${n}, dim=${DIM}, true_k=${TRUE_K}"
    serial_raw="$("$BIN" --mode serial \
                          --generate "$n" "$DIM" "$TRUE_K" \
                          --neighbors "$NEIGHBORS" \
                          --preference "$PREFERENCE" \
                          --damping "$DAMPING" \
                          --max-iter "$MAX_ITER" \
                          --tol "$TOL" \
                          --threads 1 \
                          --seed "$SEED" \
                          --verbose-every 0 \
                          --output "results/labels_n${n}_serial.csv")"
    IFS=',' read -r _mode _n _dim _neighbors _threads _build _ap serial_total _iterations _clusters _objective _edges _preference _damping _max_delta <<< "$serial_raw"
    append_result "$serial_raw" "$TRUE_K" "1.000000" "1.000000"

    for threads in $THREADS; do
        echo "Running sparse OpenMP AP: n=${n}, threads=${threads}"
        parallel_raw="$("$BIN" --mode sparse_openmp \
                                --generate "$n" "$DIM" "$TRUE_K" \
                                --neighbors "$NEIGHBORS" \
                                --preference "$PREFERENCE" \
                                --damping "$DAMPING" \
                                --max-iter "$MAX_ITER" \
                                --tol "$TOL" \
                                --threads "$threads" \
                                --seed "$SEED" \
                                --verbose-every 0 \
                                --output "results/labels_n${n}_t${threads}.csv")"
        IFS=',' read -r _mode _n _dim _neighbors _threads _build _ap parallel_total _iterations _clusters _objective _edges _preference _damping _max_delta <<< "$parallel_raw"
        speedup="$(calc_div "$serial_total" "$parallel_total")"
        efficiency="$(calc_div "$speedup" "$threads")"
        append_result "$parallel_raw" "$TRUE_K" "$speedup" "$efficiency"
    done
done

echo "Results saved to $RESULT_FILE"
