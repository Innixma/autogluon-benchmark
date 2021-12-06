#!/bin/bash

# Install code remotely
set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "Executing benchmarks..."

# AWS_ARGS=--extra_args "-m aws -p 1500"
DEFAULT_ARGS="--git_user Innixma --workspace benchmark --nohup"

BENCHMARK=ag
BRANCH="ag-2021_11_11_es"
CONSTRAINT=1h8c

# FRAMEWORK="AutoGluon_bestquality:latest"
# CUSTOM_ARGS="--framework $FRAMEWORK --benchmark $BENCHMARK --constraint $CONSTRAINT"
# $MYDIR/run_benchmark_local.sh --branch $BRANCH $CUSTOM_ARGS $DEFAULT_ARGS --extra_args "-m aws -p 1500"
#
# sleep 7200

# FRAMEWORK="AutoGluon:latest"
# CUSTOM_ARGS="--framework $FRAMEWORK --benchmark $BENCHMARK --constraint $CONSTRAINT"
# $MYDIR/run_benchmark_local.sh --branch $BRANCH $CUSTOM_ARGS $DEFAULT_ARGS --extra_args "-m aws -p 1500"
#
# sleep 7200

CONSTRAINT=4h8c

FRAMEWORK="AutoGluon_bestquality:latest"
CUSTOM_ARGS="--framework $FRAMEWORK --benchmark $BENCHMARK --constraint $CONSTRAINT"
$MYDIR/run_benchmark_local.sh --branch $BRANCH $CUSTOM_ARGS $DEFAULT_ARGS --extra_args "-m aws -p 1500"

sleep 7200

FRAMEWORK="AutoGluon:latest"
CUSTOM_ARGS="--framework $FRAMEWORK --benchmark $BENCHMARK --constraint $CONSTRAINT"
$MYDIR/run_benchmark_local.sh --branch $BRANCH $CUSTOM_ARGS $DEFAULT_ARGS --extra_args "-m aws -p 1500"

sleep 7200

echo "All benchmarks executed."
