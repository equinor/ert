#!/bin/bash
source /project/res/komodo/$KOMODO_VERSION/enable
source /opt/rh/devtoolset-7/enable
GCC_VERSION=7.3.0 CMAKE_VERSION=3.10.2 source /prog/sdpsoft/env.sh
set -e

echo "building c tests"
rm -rf build
mkdir build
pushd build
cmake .. -DBUILD_TESTS=ON\
-DENABLE_PYTHON=OFF\
-DBUILD_APPLICATIONS=ON\
-DCMAKE_INSTALL_PREFIX=install\
-DCMAKE_PREFIX_PATH=/project/res/komodo/$KOMODO_VERSION/root/\
-DCMAKE_C_FLAGS='-Werror=all'\
-DCMAKE_CXX_FLAGS='-Wno-unused-result'
make -j 12
#removing built libs in order to ensure we are using libs from komodo
rm -r lib64

echo "running ctest"
ctest --output-on-failure

popd

echo "create virtualenv"
ENV=testenv
rm -rf testenv
mkdir $ENV
python -m virtualenv --system-site-packages $ENV
source $ENV/bin/activate
python -m pip install mock decorator

echo "running pytest"
#run in a new folder so that we dont load the other python code from the source, but rather run against komodo
rm -rf tmptest
mkdir tmptest
cp -r python/tests tmptest/tests
pushd tmptest
python -m pytest \
 --ignore="tests/res/enkf/test_analysis_config.py"\
 --ignore="tests/res/enkf/test_res_config.py"\
 --ignore="tests/res/enkf/test_site_config.py"\
 --ignore="tests/res/enkf/test_workflow_list.py"


popd

