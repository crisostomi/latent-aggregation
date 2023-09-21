#!/bin/bash

# Check for the directory argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

DIRECTORY=$1

# Search for files starting with 'cache' and delete them while printing their names
find "$DIRECTORY" -type f -name 'cache*' -print -exec rm {} \;

echo "Files starting with 'cache' have been removed."
