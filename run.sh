#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run.sh — ResQue QRC nohup-safe launcher
#
# Usage:
#   bash run.sh                         # Full pipeline
#   bash run.sh --smoke_test            # Fast validation (~5 min)
#   bash run.sh --skip_download         # Use cached data
#   bash run.sh --skip_sweeps           # Skip parameter sweeps
#   bash run.sh --horizon 6             # Single horizon
#   bash run.sh --platform qbraid       # On qBraid Lab
#
# After launching, disconnect from MobaXterm freely.
# Reconnect and check progress with:
#   tail -f outputs/logs/run_*.log      # Live log
#   cat outputs/logs/nohup_out.txt      # nohup stdout/stderr
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Resolve script location ───────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Create output dirs if missing ─────────────────────────────────────────────
mkdir -p outputs/logs outputs/raw outputs/processed outputs/results outputs/figures

# ── Timestamp for this run ────────────────────────────────────────────────────
STAMP=$(date +%Y%m%d_%H%M%S)
NOHUP_OUT="outputs/logs/nohup_out_${STAMP}.txt"

# ── Python executable: prefer conda env if active ────────────────────────────
PYTHON="${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}"
PYTHON="${PYTHON:-python}"

echo "============================================================"
echo " ResQue QRC — GIC 2026"
echo " Started  : $(date)"
echo " Python   : $($PYTHON --version 2>&1)"
echo " Args     : $*"
echo " nohup log: $SCRIPT_DIR/$NOHUP_OUT"
echo "============================================================"
echo ""
echo "Launching in background. You can safely disconnect."
echo ""
echo "Monitor progress:"
echo "  tail -f $SCRIPT_DIR/outputs/logs/run_${STAMP}.log"
echo "  tail -f $SCRIPT_DIR/$NOHUP_OUT"
echo ""

# ── Launch with nohup ─────────────────────────────────────────────────────────
# PYTHONUNBUFFERED=1 ensures Python flushes stdout immediately (no buffering)
PYTHONUNBUFFERED=1 nohup "$PYTHON" -u main.py "$@" \
    > "$NOHUP_OUT" 2>&1 &

PID=$!
echo "PID: $PID  (saved to outputs/logs/run.pid)"
echo "$PID" > outputs/logs/run.pid

echo ""
echo "To stop the run:  kill \$(cat outputs/logs/run.pid)"
echo "To check if done: ps -p $PID || echo 'Run complete'"
