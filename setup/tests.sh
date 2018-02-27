#!/bin/bash

set -e

if [ -z "$VIRTUAL_ENV" ]
then
    echo "Not in venv, please activate it first."
    exit 1
fi
exclude_dirs=".svn,CVS,.bzr,.hg,.git,__pycache__,.tox,.eggs,*.egg,.ipynb_checkpoints"
flake8 --max-line-length=120 --exclude="$exclude_dirs" features/ models/ resources/ setup/
