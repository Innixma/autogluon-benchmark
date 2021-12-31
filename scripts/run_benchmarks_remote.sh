#!/bin/bash

# Install code remotely
set -e

while test $# -gt 0
do
    case "$1" in
        --remote-hostname) REMOTE_HOSTNAME="$2";;
    esac
    shift
done

if [ -z "$REMOTE_HOSTNAME" ] ; then
    echo "--remote-hostname is a required parameter (EX: --remote-hostname ec2-XX-XXX-XX-XXX.compute-1.amazonaws.com)"
    exit 1
fi

export MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Update below to indicate the EC2 instance you wish to use
# use ec2-user for AL2012 and ubuntu for Ubuntu AMI
export REMOTE_USER=ubuntu
export REMOTE_ROOT=/home/$REMOTE_USER
export REMOTE_BOX=$REMOTE_USER@$REMOTE_HOSTNAME

export REMOTE_SCRIPT_PATH=$REMOTE_ROOT/tmp/scripts

echo $REMOTE_BOX
echo $MYDIR
echo $REMOTE_SCRIPT_PATH

ssh $REMOTE_BOX mkdir -p $REMOTE_SCRIPT_PATH
rsync --delete -av $MYDIR/* $REMOTE_BOX:$REMOTE_SCRIPT_PATH/

# FIXME: Figure out how to run this with logs still in place / viewable. Maybe screen?
#  https://unix.stackexchange.com/questions/479/keep-processes-running-after-ssh-session-disconnects
ssh $REMOTE_BOX $REMOTE_SCRIPT_PATH/run_benchmarks_local_nohup.sh
