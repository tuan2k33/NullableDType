#!/bin/bash
# Build the prototype against the local NumPy dev build.
set -e
NUMPY=/mnt/c/Dev/projects/numpy/build-install/usr/lib/python3/dist-packages
PY=/mnt/c/Dev/projects/numpy/.venv/bin/python3
PYINC=$($PY -c "import sysconfig; print(sysconfig.get_paths()['include'])")
NPINC=$NUMPY/numpy/_core/include
EXT=$($PY -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

cd "$(dirname "$0")"
gcc -O2 -fPIC -shared -Wall \
    -I"$PYINC" -I"$NPINC" \
    src/nulldtype.c \
    -o "_nulldtype$EXT"
echo "built _nulldtype$EXT"
