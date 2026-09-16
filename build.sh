#!/bin/bash
# Build the extension in place, for working on it.  `pip install .` is the
# normal way to install; this is the quick loop.
#
#   PYTHON      interpreter to build for (default: python3)
#   NUMPY_SITE  directory to put in front of PYTHONPATH, for a numpy that is
#               not installed into PYTHON (e.g. a numpy dev tree)
#
# Both can also be set in a `local.env` file next to this script; it is read
# only when PYTHON is not already set.
set -e
cd "$(dirname "$0")"
# the environment wins over local.env
if [ -z "$PYTHON" ] && [ -f local.env ]; then . ./local.env; fi
PY=${PYTHON:-python3}
export PYTHONPATH=${NUMPY_SITE:+$NUMPY_SITE:}$PYTHONPATH

PYINC=$($PY -c "import sysconfig; print(sysconfig.get_paths()['include'])")
NPINC=$($PY -P -c "import numpy; print(numpy.get_include())")
EXT=$($PY -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

${CC:-gcc} -O2 -fPIC -shared -Wall $CFLAGS \
    -I"$PYINC" -I"$NPINC" \
    src/nulldtype.c \
    -o "_nulldtype$EXT"
echo "built _nulldtype$EXT against numpy $($PY -P -c 'import numpy; print(numpy.__version__)')"
