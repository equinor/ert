#!/bin/bash
set -xe
source $KOMODO_ROOT/$RELEASE_NAME/enable
source $DEVTOOL/enable

# find and check out the code that was used to build libres for this komodo relase
echo "checkout tag from komodo"
EV=$(cat $KOMODO_ROOT/$RELEASE_NAME/$RELEASE_NAME | grep "libres:" -A2 | grep "version:")
EV=($EV)    # split the string "version: vX.X.X"
EV=${EV[1]} # extract the version
EV=${EV%"+py3"}
echo "Using libres version $EV"
git checkout $EV

GCC_VERSION=7.3.0 CMAKE_VERSION=3.10.2 source $SDPSOFT/env.sh
echo "building c tests"
rm -rf build
mkdir build
pushd build
cmake .. -DBUILD_TESTS=ON\
-DENABLE_PYTHON=OFF\
-DBUILD_APPLICATIONS=ON\
-DCMAKE_INSTALL_PREFIX=install\
-DCMAKE_PREFIX_PATH=$KOMODO_ROOT/$RELEASE_NAME/root/\
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
rm -rf $ENV
mkdir $ENV
python -m virtualenv --system-site-packages $ENV
source $ENV/bin/activate
python -m pip install -r test_requirements.txt


echo "running pytest"
#run in a new folder so that we dont load the other python code from the source, but rather run against komodo
rm -rf tmptest
mkdir tmptest
cp -r python/tests tmptest/tests
pushd tmptest

export ECL_SKIP_SIGNAL=ON
python -m pytest \
 --ignore="tests/res/enkf/test_analysis_config.py"\
 --ignore="tests/res/enkf/test_res_config.py"\
 --ignore="tests/res/enkf/test_site_config.py"\
 --ignore="tests/res/enkf/test_workflow_list.py"\
 --ignore="tests/res/enkf/test_hook_manager.py"\
 --ignore="tests/legacy"
popd
