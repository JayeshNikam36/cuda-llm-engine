#!/usr/bin/env bash
set -e
docker build -t cuda-engine-dev .
docker run --rm -it --gpus all -v "$(pwd)":/workspace cuda-engine-dev "$@"
