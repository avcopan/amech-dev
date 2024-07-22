#!/usr/bin/env bash

WORK_DIR=${1}
HOST=${2}
NUM_PARALLEL=${3:-1}

MANIFEST_FILE=$(pixi info | awk -F":" '$1~/Manifest file/{gsub(/ /, "", $2); print $2;}')
echo ${MANIFEST_FILE}

WORK_DIR="$(cd "$(dirname ${WORK_DIR})"; pwd)/$(basename "${WORK_DIR}")"

function run_job() {
    local work_dir=${1}
    local manifest_file=${2}
    local num_parallel=${3}
    export PATH=${HOME}/.pixi/bin:${PATH}
    cd ${work_dir}
    echo "Running ${num_parallel} instances on $(hostname) at ${PWD}"
    which pixi
    eval "$(pixi shell-hook --manifest-path=${manifest_file})"
    which python
    which automech
    # for i in $(seq ${num_parallel}); do
    #     pixi run automech run &> "out${i}.log" &
    # done
}

# ssh ${HOST}  'bash -s' < ${SCRIPT} ${WORK_DIR}
ssh ${HOST} "$(declare -f run_job); run_job ${WORK_DIR} ${MANIFEST_FILE} ${NUM_PARALLEL}"
