#! /bin/bash
#
# Make html docs using sphinx
set -ex

cd docs
make html
cd ../
