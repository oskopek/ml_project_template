#!/bin/bash

set -e

if [ -z "$VIRTUAL_ENV" ]
then
    echo "Not in venv, please activate it first."
    exit 1
fi

nbstripout notebooks/*.ipynb
yapf -r -i features/ models/ notebooks/ resources/ setup/ -e resources/colab_utils/
