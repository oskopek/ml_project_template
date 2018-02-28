#!/bin/bash

ln -s ../../setup/pre-commit.sh .git/hooks/pre-commit
pip install jupyter_tensorboard==0.1.5
jupyter contrib nbextension install --user
jupyter nbextension enable code_prettify/code_prettify
