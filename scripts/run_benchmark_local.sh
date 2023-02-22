#!/bin/bash

# Install code remotely
set -e

while test $# -gt 0
do
    case "$1" in
        --framework) FRAMEWORK="$2";;
        --branch) BRANCH="$2";;
        --constraint) CONSTRAINT="$2";;
        --benchmark) BENCHMARK="$2";;
        --workspace) WORKSPACE_ROOT="$2";;
        --repo) REPO="$2";;
        --custom_user_dir) CUSTOM_USER_DIR=$2;;
        --git_user) GIT_USER="$2";;
        --extra_args) EXTRA_ARGS="$2";;
        --nohup) NOHUP="true";;
    esac
    shift
done

if [ -z "$FRAMEWORK" ] ; then
    echo "--framework is a required parameter (EX: --framework AutoGluon_bestquality:latest)"
    exit 1
fi

if [ -z "$BRANCH" ] ; then
    echo "--branch is a required parameter (EX: --branch ag-2021_02_24)"
    exit 1
fi

if [ -z "$CONSTRAINT" ] ; then
    echo "--constraint is a required parameter (EX: --constraint 1h8c)"
    exit 1
fi

if [ -z "$BENCHMARK" ] ; then
    echo "--benchmark is a required parameter (EX: --benchmark test)"
    exit 1
fi

if [ -z "$WORKSPACE_ROOT" ] ; then
    echo "--workspace is a required parameter (EX: --workspace tmp/results/dir)"
    exit 1
fi

if [ -z "$REPO" ] ; then
    REPO="automlbenchmark"
    echo "--repo was not specified, defaulting to: $REPO"
    # exit 1
fi

if [ -z "$CUSTOM_USER_DIR" ] ; then
    echo "--custom_user_dir is not specified, using default..."
fi

if [ -z "$GIT_USER" ] ; then
    echo "--git_user is a required parameter (EX: --git_user Innixma)"
    exit 1
fi

if [ -z "$EXTRA_ARGS" ] ; then
    echo "--extra_args was not specified, consider specifying (EX: --extra_args \"-m aws -p 1500\")"
    # exit 1
fi

echo $FRAMEWORK
echo $BRANCH
echo $CONSTRAINT
echo $BENCHMARK
echo $WORKSPACE_ROOT
echo $REPO
echo $CUSTOM_USER_DIR
echo $GIT_USER
echo $EXTRA_ARGS

WORKSPACE=$WORKSPACE_ROOT/$BRANCH
REPO_ROOT=$WORKSPACE/$REPO
rm -rf $REPO_ROOT/
mkdir -p $WORKSPACE
cd $WORKSPACE/

git clone -b $BRANCH https://github.com/$GIT_USER/$REPO
cd $REPO
git fetch
git pull
cd ..
python3 -m venv venv
source venv/bin/activate
pip3 install -U pip
pip3 install -U setuptools wheel
pip3 install -r $REPO/requirements.txt

mkdir -p run
cd run

if [ ! -z "$CUSTOM_USER_DIR" ] ; then
    CUSTOM_USER_DIR_OG="../${REPO}/${CUSTOM_USER_DIR}"
    CUSTOM_USER_DIR_NEW="${CUSTOM_USER_DIR}"
    CUSTOM_USER_DIR="-u ${CUSTOM_USER_DIR_NEW}"
    mkdir -p ${CUSTOM_USER_DIR_NEW}
    cp -r ${CUSTOM_USER_DIR_OG} .
fi

echo "==================================="
echo "Preparing to run framework $FRAMEWORK ..."
echo "Working directory: $PWD"
COMMAND_1="python ../$REPO/runbenchmark.py $FRAMEWORK $BENCHMARK $CONSTRAINT $CUSTOM_USER_DIR $EXTRA_ARGS"
echo "Commands to run:"
echo $COMMAND_1

echo "Executing commands for $FRAMEWORK"
if [ -z "$NOHUP" ] ; then
  $COMMAND_1
else
  LOG_FILE_NAME="log_${FRAMEWORK}_${BENCHMARK}_${CONSTRAINT}.file"
  LOG_FILE_NAME=$(echo "$LOG_FILE_NAME" | tr '/\' _)  # Remove / and \
  nohup $COMMAND_1 > LOG_FILE_NAME 2>&1 &
fi
echo "Commands executed for $FRAMEWORK"
echo "==================================="

exit 0
