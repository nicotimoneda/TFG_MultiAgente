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
ENABLE_MBPP="${ENABLE_MBPP:-0}"      # set ENABLE_MBPP=1 to add MBPP phase
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

ollama_alive() {
    curl -sf -m 3 http://localhost:11434/api/tags > /dev/null 2>&1
}

ensure_ollama() {
    if ollama_alive; then
        return 0
    fi
    log "ollama is not responding; restarting"
    pkill -f "ollama serve" 2>/dev/null
    sleep 2
    nohup ollama serve > "$OLLAMA_LOG" 2>&1 &
    # wait up to 30s for it to come up
    for i in $(seq 1 30); do
        if ollama_alive; then
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
        ensure_ollama || { sleep "$SLEEP_RESTART"; continue; }

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

            log "Runner not running (completed=$completed, target=$expected_total). Launching (restart #$restarts)."
            LLM_BACKEND=ollama nohup python experiments/run_experiments.py \
                --model "$MODEL" \
                --configs "$configs" \
                --benchmarks "$benchmarks" \
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

# Phase 1: main matrix (already underway).
# expected_total for phase 1: 164 * 3 * 5 = 2 460
PHASE1_CONFIGS="baseline,sequential,self_reflection_r1,self_reflection_r2,self_reflection_r3"
PHASE1_TOTAL=2460

# Phase 2: ablations.
# expected_total for phase 2: phase 1 total + 164 * 3 * 3 = 2 460 + 1 476 = 3 936
PHASE2_CONFIGS="ablation_no_pm,ablation_no_architect,ablation_no_reviewer"
PHASE2_TOTAL=3936

# Phase 3 (optional): MBPP — same configs as 1+2 on 200 MBPP.
# expected_total for phase 3: 3 936 + 200 * 3 * 8 = 3 936 + 4 800 = 8 736
PHASE3_CONFIGS="$PHASE1_CONFIGS,$PHASE2_CONFIGS"
PHASE3_TOTAL=8736

log "Watchdog started (pid=$$, log=$WATCHDOG_LOG)"
log "MODEL=$MODEL  ENABLE_MBPP=$ENABLE_MBPP"

run_phase "main"      "$PHASE1_CONFIGS" "humaneval"        "$PHASE1_TOTAL"  || exit 1
run_phase "ablations" "$PHASE2_CONFIGS" "humaneval"        "$PHASE2_TOTAL"  || exit 1

if [ "$ENABLE_MBPP" = "1" ]; then
    run_phase "mbpp"  "$PHASE3_CONFIGS" "humaneval,mbpp"   "$PHASE3_TOTAL"  || exit 1
fi

log "=== ALL ENABLED PHASES COMPLETE ==="

# Post-processing: regenerate figures, tables and adherence summary.
log "Running analyze_results.py + adherence_metric.py"
LLM_BACKEND=ollama python experiments/analyze_results.py >> "$WATCHDOG_LOG" 2>&1
LLM_BACKEND=ollama python experiments/adherence_metric.py >> "$WATCHDOG_LOG" 2>&1

log "Watchdog exit OK."
rm -f "$WATCHDOG_PID"
