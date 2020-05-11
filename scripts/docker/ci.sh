#/bin/bash
# 
# Local tests of CI jobs
# Run this from CI job docker container
set -ex

echo 'Running black'
black --check merf

echo 'Running flake'
flake8 merf

echo 'Running pytest'
python merf/merf_test.py -v

echo 'Finished tests'
