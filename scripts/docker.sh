#!/usr/bin/env bash
set -e
docker build -t cuda-engine-dev .
docker run --rm -it --gpus all -v "$(pwd)":/workspace -v "$HOME/models":/models:ro cuda-engine-dev "$@"
