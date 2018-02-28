#!/bin/bash

set -e

if [ -z "$VIRTUAL_ENV" ]
then
    echo "Not in venv, please activate it first."
    exit 1
fi
echo "Running flake8..."
flake8 --statistics .
echo -e "\nFLAKE8: PASSED"
