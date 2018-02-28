#!/bin/bash

# Install the pre-commit hook
rm -f .git/hooks/pre-commit
ln -s ../../setup/pre-commit.sh .git/hooks/pre-commit

# Install jupyter_tensorboard
pip install jupyter_tensorboard==0.1.5

# Install code_prettify
jupyter contrib nbextension install --user
jupyter nbextension enable code_prettify/code_prettify

# Install nbstripout hooks
git config filter.nbstripout.clean 'nbstripout'
git config filter.nbstripout.smudge cat
git config filter.nbstripout.required true

git config diff.ipynb.textconv 'nbstripout -t'
