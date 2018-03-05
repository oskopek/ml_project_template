#!/bin/bash

set -e

if ./setup/check_venv.sh
then
    echo "Not in venv, please activate it first."
    exit 1
fi

nbstripout notebooks/*.ipynb
python setup/yapf_nbformat.py notebooks/*.ipynb
yapf -r -i . -e resources/colab_utils/ -e venv/
