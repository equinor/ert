#!/bin/bash

source /project/res/komodo/$KOMODO_VERSION/enable
ENV=testenv
rm -rf testenv
mkdir $ENV
python -m virtualenv --system-site-packages $ENV
source $ENV/bin/activate
python -m pip install mock

#ensure that we are not using code from git, but rather from komodo
ls|grep -xFv tests|grep -xFv testkomodo.sh|grep -xFv testenv|grep -xFv test-data|xargs rm -r

ls

echo "running pytest"
python -m pytest
