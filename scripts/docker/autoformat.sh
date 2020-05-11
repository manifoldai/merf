#/bin/bash
#
# autoformat.sh
# 
# Runs all autoformaters
# Run this from CI job docker container
set -ex

echo 'Running isort'
isort -rc merf

echo 'Running black'
black merf

echo 'Finished auto formatting'
