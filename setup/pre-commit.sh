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

# If you want to allow non-ASCII filenames set this variable to true.
allownonascii=$(git config --bool hooks.allownonascii)

# Redirect output to stderr.
exec 1>&2

echo "Running pre-commit hook..."

# Cross platform projects tend to avoid non-ASCII filenames; prevent
# them from being added to the repository. We exploit the fact that the
# printable range starts at the space character and ends with tilde.
if [ "$allownonascii" != "true" ] &&
    # Note that the use of brackets around a tr range is ok here, (it's
    # even required, for portability to Solaris 10's /usr/bin/tr), since
    # the square bracket bytes happen to fall in the designated range.
    test $(git diff --cached --name-only --diff-filter=A -z $against |
      LC_ALL=C tr -d '[ -~]\0' | wc -c) != 0
then
    cat <<\EOF
Error: Attempt to add a non-ASCII file name.

This can cause problems if you want to work with people on other platforms.

To be portable it is advisable to rename the file.

If you know what you are doing you can disable this check using:

  git config hooks.allownonascii true
EOF
    exit 1
fi

./setup/tests.sh
RESULT=$?
[ $RESULT -ne 0 ] && exit 1

cleaningtest || exit 1

echo -e "\nPre-commit hooks PASSED"
