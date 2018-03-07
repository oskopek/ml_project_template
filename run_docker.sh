#!/bin/bash
img="registry.gitlab.com/oskopek/ml/cpu"

set -e

pass="$1"
if [ -z $pass ]; then
    echo "Please supply a Jupyter notebook password as the first parameter."
    exit 1
fi

set -- "${@:2}"

cmd="sudo docker run -it -e PASSWORD=$pass -p 8888:8888 -p 6006:6006 $img -- $@"
echo "Running: $cmd"
eval "$cmd"
