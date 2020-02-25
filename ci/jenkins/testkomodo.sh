#!/bin/bash
set -xe
PROJECT=libres
RELEASE_PATH=${KOMODO_ROOT}/${RELEASE_NAME}
GIT=${SDPSOFT}/bin/git
source $KOMODO_ROOT/$RELEASE_NAME/enable
source $DEVTOOL/enable

# find and check out the code that was used to build libres for this komodo relase
if [[ -z "${sha1// }" ]]; then
    # this is not a PR build, the komodo everest verison is checked out
    EV=$(cat ${RELEASE_PATH}/${RELEASE_NAME} | grep "${PROJECT}:" -A2 | grep "version:")
    EV=($EV)    # split the string "version: vX.X.X"
    EV=${EV[1]} # extract the version
    EV=${EV%"+py3"}
    echo "Using ${PROJECT} version ${EV}"
    $GIT checkout $EV
fi

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
if [[ -z "${sha1// }" ]]; then
    rm -r lib64
fi
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
if [[ -z "${sha1// }" ]]; then
    rm -rf tmptest
    mkdir tmptest
    cp -r python/tests tmptest/tests
    pushd tmptest
fi

export ECL_SKIP_SIGNAL=ON
python -m pytest \
 --ignore="tests/res/enkf/test_analysis_config.py"\
 --ignore="tests/res/enkf/test_res_config.py"\
 --ignore="tests/res/enkf/test_site_config.py"\
 --ignore="tests/res/enkf/test_workflow_list.py"\
 --ignore="tests/res/enkf/test_hook_manager.py"\
 --ignore="tests/legacy"
