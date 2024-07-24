#!/usr/bin/env bash

RUN_DIR=${1}
PIXI_MANIFEST=${2}
cd ${RUN_DIR}
echo "Hello from ${PWD} on $(hostname)"
eval "$(pixi shell-hook --manifest-path=${PIXI_MANIFEST})"
automech run &> out.log
