#!/bin/bash

function cleaningtest {
    echo -e "\nPython cleaning test..."
    stagedfiles=$(git diff --cached --name-only | grep '.py$')
    [ -z "$stagedfiles" ] && return 0

    unformatted=$(yapf --diff $stagedfiles | grep "(reformatted)" | awk '{print $2}')
    [ -z "$unformatted" ] && return 0

    echo "The following files are not formatted properly:"
    for fn in $unformatted; do
        echo "    $fn"
    done

    echo -e "\nPython files must be formatted with YAPF. Please run:"
    echo "    ./setup/clean.sh"
    return 1
}

# Redirect output to stderr.
exec 1>&2

echo "Running pre-commit hook..."

./setup/tests.sh
RESULT=$?
[ $RESULT -ne 0 ] && exit 1

cleaningtest || exit 1

echo -e "\nPre-commit hooks PASSED"
