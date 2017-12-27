#!/bin/bash

set -e

venv="venv"

mkdir "$venv"
python3 -m pip install virtualenv
python3 -m virtualenv "$venv" --no-site-packages
source "$venv"/bin/activate
pip install -r "requirements.txt"
