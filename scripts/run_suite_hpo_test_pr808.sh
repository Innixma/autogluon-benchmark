#!/bin/bash

####################
# Install code
set -e

VENV_NAME="autogluon_test"
WORKSPACE="workspace"

mkdir -p ~/virtual
python3 -m venv ~/virtual/$VENV_NAME
source ~/virtual/$VENV_NAME/bin/activate
mkdir $WORKSPACE && cd $WORKSPACE

git clone https://github.com/awslabs/autogluon
pip install --upgrade pip
pip install --upgrade setuptools
pip install --upgrade "mxnet<2.0.0"
cd autogluon && ./full_install.sh
cd ..

git clone https://github.com/Innixma/autogluon-benchmark
cd autogluon-benchmark && pip install -e .
cd ..


####################
# Run mainline

python autogluon-benchmark/examples/train_suite_hpo_small.py

echo "Mainline finished"

# Run PR 808

cd autogluon
git remote add RuohanW https://github.com/RuohanW/autogluon
git fetch RuohanW
git checkout --track RuohanW/hyperband_dt
cd ..
python autogluon-benchmark/examples/train_suite_hpo_small.py

echo "PR 808 finished"

####################
# Revert to master
cd autogluon
git checkout origin/master
cd ..

exit 0
