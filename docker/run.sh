#!/bin/bash

set -e

arg="$1"
if [ -z "$arg" ]; then
    echo 'No run config passed. Can either be a run config file, "bash", or "jupyter"'
    exit 1
fi

# Activate virtual env
source venv/bin/activate

if [ -z "$VIRTUAL_ENV" ]
then
    echo "Not in venv, please activate it first."
    exit 1
fi

# Run jupyter notebooks
if [ "$arg" == 'jupyter' ]; then
    exec jupyter notebook --allow-root
fi

# Run bash
if [ "$arg" == 'bash' ]; then
    exec bash
fi

if [ "$arg" == 'notebook' ]; then
    exec jupyter nbconvert --to notebook --stdout --execute $@
fi

# Run a specific configuration
exec python -m docker.run_config $@ # Pass arg and all successive arguments
