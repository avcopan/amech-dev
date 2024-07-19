#!/usr/bin/env bash

WORK_DIR=${1}
HOST=${2}
NUM_PARALLEL=${3:-1}

WORK_DIR="$(cd "$(dirname ${WORK_DIR})"; pwd)/$(basename "${WORK_DIR}")"

function run_job() {
    local work_dir=${1}
    local num_parallel=${2:-1}
    cd ${work_dir}
    echo "Running ${num_parallel} instances on $(hostname) at ${PWD}"
    source ~/.bash_profile
    for i in $(seq ${num_parallel}); do
        pixi run automech run &> "out${i}.log" &
    done
}

# ssh ${HOST}  'bash -s' < ${SCRIPT} ${WORK_DIR}
ssh ${HOST} "$(declare -f run_job); run_job ${WORK_DIR} ${NUM_PARALLEL}"
