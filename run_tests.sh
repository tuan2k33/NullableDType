#!/bin/bash
set -e
cd "$(dirname "$0")"
./build.sh
NUMPY=/mnt/c/Dev/projects/numpy/build-install/usr/lib/python3/dist-packages
PYTHONPATH=$NUMPY:. /mnt/c/Dev/projects/numpy/.venv/bin/python3 -P -m pytest test_basic.py -q --no-header
