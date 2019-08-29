#!/bin/bash

# This script is designed to be able to run in stages in the Jenkins pipline
#
# You can also run it locally. Note that only the committed code is tested,
# not changes in your working directory.
#
# If you want to run the whole build in one go:
#   sh testjenkins.sh build_and_test
#
# If you want to run the build in separate stages:
#   sh testjenkins.sh setup
#   sh testjenkins.sh build_elc
#   etc...
#
# By default it will try to build everything inside a folder 'jenkinsbuild'
# You can override this with the env variable WORKING_DIR e.g.:
#   WORKING_DIR=$(mktemp -d) sh testjenkins.sh build_and_test

build_and_test () {
	rm -rf jenkinsbuild
	mkdir jenkinsbuild
	run setup
	run build_ecl
	run build_res
	run run_ctest
	run run_pytest_equinor
	run run_pytest_normal
}

setup () {
	run source_build_tools
	run setup_variables
	run clone_repos
	run create_directories
	run create_virtualenv
}

build_ecl () {
	run enable_environment

	pushd $LIBECL_ROOT
	python -m pip install -r requirements.txt
	pushd $LIBECL_BUILD

	cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALL \
		  -DINSTALL_ERT_LEGACY=ON \
		  -DBUILD_TESTS=OFF \
		  -DUSE_RPATH=ON \
		  -DENABLE_PYTHON=ON
	make -j 6 install
	popd
	popd
}

build_res () {
	run enable_environment

	pushd $LIBRES_ROOT
	python -m pip install -r requirements.txt
	pushd $LIBRES_BUILD
	echo "PYTHON:"$(which python)
	cmake .. -DEQUINOR_TESTDATA_ROOT=/project/res-testdata/ErtTestData \
		  -DINSTALL_ERT_LEGACY=ON \
		  -DCMAKE_PREFIX_PATH=$INSTALL \
		  -DCMAKE_MODULE_PATH=$INSTALL/share/cmake/Modules \
		  -DCMAKE_INSTALL_PREFIX=$INSTALL \
		  -DBUILD_TESTS=ON \
		  -DENABLE_PYTHON=ON

	make -j 6 install
	popd
	popd
}

source_build_tools() {
	source /opt/rh/devtoolset-7/enable
	LIBECL_VERSION="" PYTHON_VERSION="2.7.14" GCC_VERSION=7.3.0 CMAKE_VERSION=3.10.2 source /prog/sdpsoft/env.sh
	python --version
	gcc --version
	cmake --version

	set -e
}

setup_variables () {

	ENV=$WORKING_DIR/venv
	INSTALL=$WORKING_DIR/install

	LIBECL_ROOT=$WORKING_DIR/libecl

	LIBECL_BUILD=$LIBECL_ROOT/build

	LIBRES_ROOT=$WORKING_DIR/libres
	LIBRES_BUILD=$LIBRES_ROOT/build

	KOMODO_VERSION=bleeding
}

enable_environment () {
	run source_build_tools
	run setup_variables

	source $ENV/bin/activate
	export ERT_SHOW_BACKTRACE=Y
	export ECL_SITE_CONFIG=/project/res/komodo/$KOMODO_VERSION/root/lib/python2.7/site-packages/res/fm/ecl/ecl_config.yml
	export RMS_SITE_CONFIG=/project/res/komodo/$KOMODO_VERSION/root/lib/python2.7/site-packages/res/fm/rms/rms_config.yml
	export LD_LIBRARY_PATH=$INSTALL/lib64:$LD_LIBRARY_PATH
	export PYTHONPATH=$INSTALL/lib/python2.7/site-packages:$PYTHONPATH
	export PYTHONFAULTHANDLER=PYTHONFAULTHANDLER
}

create_directories () {
	mkdir $INSTALL
	mkdir $LIBECL_BUILD
	mkdir $LIBRES_BUILD
}

clone_repos () {
	echo "Cloning into $LIBRES_ROOT"
	git clone . $LIBRES_ROOT

	source ./.libecl_version

	echo "Cloning into $LIBECL_ROOT"
	git clone -b $LIBECL_VERSION https://github.com/equinor/libecl $LIBECL_ROOT
}

create_virtualenv () {
	mkdir $ENV
	python -m virtualenv $ENV
	source $ENV/bin/activate
	python -m pip install pytest faulthandler decorator mock
}

run_ctest () {
	run enable_environment
	pushd $LIBRES_BUILD
	ctest -j $CTEST_JARG -E Lint --output-on-failure
	popd
}

run_pytest_normal () {
	run enable_environment
	pushd $LIBRES_ROOT/python
	python -m pytest -s -m "not equinor_test"
	popd
}


run_pytest_equinor () {
	run enable_environment
	pushd $LIBRES_ROOT/python
	python -m pytest -s -m "equinor_test"
	popd
}


run () {
	echo ""
	echo "----- Running step: $1 -----"
	echo ""
	$1

	echo ""
	echo "----- $1 finished -----"
	echo ""
}

if [ -z "$CTEST_JARG" ]
	then
		CTEST_JARG="6"
fi

if [ -z "$WORKING_DIR" ]
	then
		WORKING_DIR=$(pwd)/jenkinsbuild
fi

if [ $# -ne 0 ]
	then
		run $1
fi
