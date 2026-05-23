#!/usr/bin/env bash
#
# experiment_watchdog.sh — keep the experimental pipeline alive and
# automatically advance through the planned phases.
#
# Phases:
#   1. Main matrix:    5 configs × 3 seeds × 164 HumanEval (2 460 runs).
#   2. Ablations:      3 ablation configs × 3 seeds × 164 HumanEval (1 476 runs).
#   3. MBPP (optional, off by default).
#
# Each phase invokes experiments/run_experiments.py, which is itself resumable;
# the watchdog only needs to handle "process crashed" and "phase finished".
#
# Behaviour:
#   - Restarts `ollama serve` if it is not running.
#   - Restarts the runner if it dies but the phase isn't complete (progress.json
#     shows completed < expected for that phase).
#   - Advances to the next phase when the current one completes cleanly.
#   - Exits 0 when all enabled phases are complete.
#   - Logs everything to experiments/logs/watchdog.log.
#
# Designed for `nohup ... &` invocation:
#   cd ~/UNI/TFG/TFG_MultiAgente
#   nohup bash scripts/experiment_watchdog.sh > experiments/logs/watchdog.out 2>&1 &
#
# Cancel cleanly with:  kill $(cat experiments/logs/watchdog.pid)

set -u  # no -e because we want to keep going through failures

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL="${MODEL:-qwen2.5-coder:7b-instruct-q4_K_M}"
LLM_BACKEND_VAR="${LLM_BACKEND_VAR:-ollama}"
WORKERS="${WORKERS:-1}"
ENABLE_MBPP="${ENABLE_MBPP:-0}"
ENABLE_SR_R2="${ENABLE_SR_R2:-0}"    # off: budget-driven scope reduction
ENABLE_SR_R3="${ENABLE_SR_R3:-0}"    # off: marginal over r2
ABLATION_SUBSET_SIZE="${ABLATION_SUBSET_SIZE:-0}"   # 0 disables ablation phase
SLEEP_OK=120                          # seconds between healthy checks
SLEEP_RESTART=15                      # seconds before restarting after a crash
MAX_RESTARTS_PER_PHASE=100            # circuit breaker

LOG_DIR="experiments/logs"
mkdir -p "$LOG_DIR"
WATCHDOG_LOG="$LOG_DIR/watchdog.log"
WATCHDOG_PID="$LOG_DIR/watchdog.pid"
RUNNER_LOG="$LOG_DIR/run.out"
OLLAMA_LOG="$LOG_DIR/ollama.out"

echo $$ > "$WATCHDOG_PID"

log() {
    local ts; ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "[$ts] $*" | tee -a "$WATCHDOG_LOG"
}

# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------

ensure_backend() {
    if [ "$LLM_BACKEND_VAR" = "cerebras" ]; then
        # Cerebras is a remote API; nothing to keep alive locally beyond
        # the .env-supplied API key. A failed call is caught by the
        # runner's per-row try/except, not by the watchdog.
        return 0
    fi
    # Legacy Ollama path (kept for reference)
    if curl -sf -m 3 http://localhost:11434/api/tags > /dev/null 2>&1; then
        return 0
    fi
    log "ollama is not responding; restarting"
    pkill -f "ollama serve" 2>/dev/null
    sleep 2
    nohup ollama serve > "$OLLAMA_LOG" 2>&1 &
    for i in $(seq 1 30); do
        if curl -sf -m 3 http://localhost:11434/api/tags > /dev/null 2>&1; then
            log "ollama is back (waited ${i}s)"
            return 0
        fi
        sleep 1
    done
    log "ERROR: ollama did not come back within 30s"
    return 1
}

runner_pid() {
    pgrep -f "python experiments/run_experiments.py" | head -1
}

runner_alive() {
    [ -n "$(runner_pid)" ]
}

progress_field() {
    # $1 = key (e.g. "completed", "total")
    python3 -c "
import json, sys
try:
    p = json.load(open('experiments/results/progress.json'))
    print(p.get('$1', ''))
except Exception:
    print('')
"
}

# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------

run_phase() {
    local phase_name="$1"
    local configs="$2"
    local benchmarks="$3"
    local expected_total="$4"

    log "=== PHASE START: $phase_name (target $expected_total runs) ==="

    local restarts=0
    while true; do
        ensure_backend || { sleep "$SLEEP_RESTART"; continue; }

        if ! runner_alive; then
            # Check if the phase is already done
            local completed
            completed=$(progress_field completed)
            local total
            total=$(progress_field total)
            if [ -n "$completed" ] && [ -n "$total" ] && [ "$completed" -ge "$expected_total" ] 2>/dev/null; then
                log "Phase '$phase_name' completed: $completed/$expected_total."
                return 0
            fi

            if [ "$restarts" -ge "$MAX_RESTARTS_PER_PHASE" ]; then
                log "ERROR: max restarts ($MAX_RESTARTS_PER_PHASE) reached for phase '$phase_name'. Aborting."
                return 1
            fi

            log "Runner not running (completed=$completed, target=$expected_total). Launching (restart #$restarts, workers=$WORKERS, backend=$LLM_BACKEND_VAR, model=$MODEL)."
            LLM_BACKEND="$LLM_BACKEND_VAR" nohup python experiments/run_experiments.py \
                --model "$MODEL" \
                --configs "$configs" \
                --benchmarks "$benchmarks" \
                --workers "$WORKERS" \
                >> "$RUNNER_LOG" 2>&1 &
            restarts=$((restarts + 1))
            sleep "$SLEEP_RESTART"
            continue
        fi

        # Healthy path
        sleep "$SLEEP_OK"
    done
}

# ---------------------------------------------------------------------------
# Phase plan
# ---------------------------------------------------------------------------

# Phase 1: main matrix.
# Default: 4 configs (baseline + sequential + SR_r1 + SR_r2). SR_r3 disabled
# by default because chapter 8.5 documents r2 as the practical ceiling on the
# 7B model under HumanEval and the marginal value of r3 does not justify the
# additional ~5 days of compute. Re-enable with ENABLE_SR_R3=1.
if [ "$ENABLE_SR_R3" = "1" ]; then
    PHASE1_CONFIGS="baseline,sequential,self_reflection_r1,self_reflection_r2,self_reflection_r3"
    PHASE1_TOTAL=2460
elif [ "$ENABLE_SR_R2" = "1" ]; then
    PHASE1_CONFIGS="baseline,sequential,self_reflection_r1,self_reflection_r2"
    PHASE1_TOTAL=1968
else
    PHASE1_CONFIGS="baseline,sequential,self_reflection_r1"
    PHASE1_TOTAL=1476   # 3 configs * 164 problems * 3 seeds — finishes ~June 1
fi

# Phase 2: ablations on a problem subset (default 50 problems) to keep total
# compute under control while still answering the role-contribution question.
PHASE2_CONFIGS="ablation_no_pm,ablation_no_architect,ablation_no_reviewer"
# Each ablation runs against ABLATION_SUBSET_SIZE problems × 3 seeds.
# Total added = 3 configs * subset * 3 seeds.
PHASE2_TOTAL=$(( PHASE1_TOTAL + 3 * ABLATION_SUBSET_SIZE * 3 ))

log "Watchdog started (pid=$$, log=$WATCHDOG_LOG)"
log "BACKEND=$LLM_BACKEND_VAR  MODEL=$MODEL  WORKERS=$WORKERS  ENABLE_MBPP=$ENABLE_MBPP"

run_phase "main"      "$PHASE1_CONFIGS" "humaneval"        "$PHASE1_TOTAL"  || exit 1
# Ablation phase runs on a subset; the runner respects --problems if we passed
# it, but here we keep the default and let the user run a subset variant by
# editing this script when phase 2 is reached. Disable phase 2 by setting
# ABLATION_SUBSET_SIZE=0.
if [ "$ABLATION_SUBSET_SIZE" -gt 0 ]; then
    run_phase "ablations" "$PHASE2_CONFIGS" "humaneval"    "$PHASE2_TOTAL"  || exit 1
fi

log "=== ALL ENABLED PHASES COMPLETE ==="

# Post-processing: regenerate figures, tables and adherence summary.
log "Running analyze_results.py + adherence_metric.py"
LLM_BACKEND="$LLM_BACKEND_VAR" python experiments/analyze_results.py >> "$WATCHDOG_LOG" 2>&1
LLM_BACKEND="$LLM_BACKEND_VAR" python experiments/adherence_metric.py >> "$WATCHDOG_LOG" 2>&1

log "Watchdog exit OK."
rm -f "$WATCHDOG_PID"
