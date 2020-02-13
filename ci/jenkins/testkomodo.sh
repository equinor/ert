#!/bin/bash
PROJECT="ert"
RELEASE_PATH=${KOMODO_ROOT}/${RELEASE_NAME}
echo "unit testing ${PROJECT} against ${RELEASE_NAME}"

source ${RELEASE_PATH}/enable
source ${DEVTOOL}/enable
GCC_VERSION=7.3.0 CMAKE_VERSION=3.10.2 source ${SDPSOFT}/env.sh

GIT=${SDPSOFT}/bin/git

EV=$(cat ${RELEASE_PATH}/${RELEASE_NAME} | grep "${PROJECT}:" -A2 | grep "version:")
EV=($EV)    # split the string "version: vX.X.X"
EV=${EV[1]} # extract the version
EV=${EV%"+py3"}
echo "Using ${PROJECT} version ${EV}"
$GIT checkout $EV

echo "Creating virtualenv"
ENV="testenv"
rm -rf $ENV
mkdir $ENV
python -m virtualenv --system-site-packages $ENV
source $ENV/bin/activate
python -m pip install -r dev-requirements.txt


echo "Running pytest"
# Run in a new folder so that we dont load the other python code from the source, but rather run against komodo
rm -rf tmptest
mkdir tmptest
cp -r tests tmptest/tests
cp -r test-data tmptest/test-data
pushd tmptest
python -m pytest
popd
