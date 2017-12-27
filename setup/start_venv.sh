#!/bin/bash

set -e

venv="venv"
ip="192.168.70.179"

source "$venv"/bin/activate
jupyter lab --no-browser --ip "$ip" --port 8888
