#!/bin/bash

set -e

prefix="resources/google_drive_upload"

folder="notebooks"
script="$prefix/drive.py"
folderId="`cat $prefix/folderId.txt`"

python $script $folder $folderId $prefix/client_secret.json

