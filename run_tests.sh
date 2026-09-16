#!/bin/bash
# Build in place, then run the suite.  Extra arguments go to pytest.
# See build.sh for PYTHON, NUMPY_SITE and local.env.
set -e
cd "$(dirname "$0")"
./build.sh
# the environment wins over local.env
if [ -z "$PYTHON" ] && [ -f local.env ]; then . ./local.env; fi
PY=${PYTHON:-python3}
PYTHONPATH=${NUMPY_SITE:+$NUMPY_SITE:}.:$PYTHONPATH $PY -P -m pytest test_basic.py -q --no-header "$@"
