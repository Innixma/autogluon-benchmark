#!/bin/bash

# Install code remotely
set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "Executing benchmarks (nohup)..."

DATETIME=$(date +"%Y_%m_%d_%H_%M_%S")
LOG_FILE_NAME="benchmark_log_$DATETIME.file"
LOG_FILE_PATH="$MYDIR/$LOG_FILE_NAME"

nohup "$MYDIR/run_benchmarks_local.sh" > "$LOG_FILE_PATH" 2>&1 &

echo "Benchmarks executed (nohup)."
echo "Tailing log file... Log file can be found at $LOG_FILE_PATH"

tail -f "$LOG_FILE_PATH"
