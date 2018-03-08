#!/bin/bash

set -e

prefix="resources/google_drive_upload"

notebook_folder="notebooks"
drive_script="$prefix/drive.py"
folder_id="`cat $prefix/folderId.txt`"
client_secret="$prefix/client_secret.json"

python "$drive_script" "$notebook_folder" "$folder_id" "$client_secret"

