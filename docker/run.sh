#!/bin/bash

set -e

# Replace docker args with the script args
args="$(echo $@ | sed 's/^.*-- //')"
echo "Args: $args" >&2
set -- $args

arg="$1"
if [ -z "$arg" ]; then
    echo 'No run config passed. Can either be a run config file, "jupyter", or "notebook"'
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

if [ "$arg" == 'notebook' ]; then
    exec jupyter nbconvert --to notebook --stdout --execute $@
fi

# Run a specific configuration + TensorBoard
tensorboard --logdir data_out >/dev/null & # Run tensorboard in the background
exec python -m docker.run_config $@ # Pass arg and all successive arguments
