#!/bin/bash

set -e

export VENV=ag_tmp
export VENV_LOCATION="~/virtual"

mkdir -p $VENV_LOCATION
python3 -m venv $VENV_LOCATION/$VENV
source $VENV_LOCATION/$VENV/bin/activate
pip install -U pip
pip install -U setuptools wheel
pip install -U "mxnet<2.0.0"
pip install autogluon

exit 0
