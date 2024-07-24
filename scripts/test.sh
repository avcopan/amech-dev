#!/usr/bin/env bash

JOB_MEM=${1}        # memory required per job
JOB_NPROCS=${2}     # number of cores required per job
IFS="," read -ra DIRS <<< "${3}"    # list of run directories
IFS="," read -ra NODES <<< "${4}"   # list of nodes for running

echo "JOB_MEM = ${JOB_MEM}"
echo "JOB_NPROCS = ${JOB_NPROCS}"

echo "DIRS:"
echo ${DIRS[*]}

echo "NODES:"
SSHLOGIN=$(IFS=,; echo "${NODES[*]}")
echo $SSHLOGIN

run() {
    local run_dir=${1}
    cd ${run_dir}
    echo "Hello from ${PWD} on $(hostname)"
}
export -f run

# parallel --sshlogin $SSHLOGIN "$(declare -f run); run {}" ::: ${DIRS[*]}
# parallel --sshlogin $SSHLOGIN "cd {} && echo "Hello from ${PWD} on $(hostname)"" ::: ${DIRS[*]}
parallel --env run --sshlogin $SSHLOGIN "run" ::: ${DIRS[*]}

# parallel --sshlogin csed-0009,csed-0010 "cd ${PWD} && ./task.sh" ::: 1 2 3 4 5
