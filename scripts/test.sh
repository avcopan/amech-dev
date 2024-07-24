#!/usr/bin/env bash

JOB_MEM=${1}        # memory required per job
JOB_NPROCS=${2}     # number of cores required per job
IFS="," read -ra DIRS <<< "${3}"    # list of run directories
IFS="," read -ra NODES <<< "${4}"   # list of nodes for running

echo "JOB_MEM = ${JOB_MEM}"
echo "JOB_NPROCS = ${JOB_NPROCS}"

echo "DIRS:"
echo ${DIRS[*]}

# Determine how many workers to put on each node, based on job memory and nprocs
echo "Determining node capacities:"
SSHLOGINS=()
SSHLOGIN=""
for node in "${NODES[@]}"; do
    node_mem_kb=$(ssh ${node} "grep MemTotal /proc/meminfo" | awk '{print $2}')
    node_mem=$((node_mem_kb / 1000000))
    node_nprocs=$(ssh ${node} "nproc --all")
    node_cap1=$((node_mem / JOB_MEM))
    node_cap2=$((node_nprocs / JOB_NPROCS))
    node_nwork=$((node_cap1 < node_cap2 ? node_cap1 : node_cap2))
    echo "Node ${node}: Memory=${node_mem} | Nprocs=${node_nprocs} | NWorkers=${node_nwork}"
    SSHLOGINS+=("${node_nwork}/${node}")
done
SSHLOGIN=$(IFS=,; echo "${SSHLOGINS[*]}")
echo "Running with --sshlogin ${SSHLOGIN}"

run() {
    local run_dir=${1}
    cd ${run_dir}
    echo "Hello from ${PWD} on $(hostname)"
}
export -f run

# parallel --sshlogin $SSHLOGIN "$(declare -f run); run {}" ::: ${DIRS[*]}
# parallel --sshlogin $SSHLOGIN "cd {} && echo "Hello from ${PWD} on $(hostname)"" ::: ${DIRS[*]}
# parallel --sshlogin $SSHLOGIN "cd ${PWD} && ./scripts/_run.sh" ::: ${DIRS[*]}
parallel --sshlogin ${SSHLOGIN} "cd ${PWD} && ./scripts/_run.sh" ::: ${DIRS[*]}
# parallel --eta --sshlogin 4/csed-0009,3/csed-0010 "cd ${PWD} && ./scripts/_run.sh" ::: ${DIRS[*]}

# parallel --sshlogin csed-0009,csed-0010 "cd ${PWD} && ./task.sh" ::: 1 2 3 4 5
