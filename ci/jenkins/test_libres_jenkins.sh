#!/bin/bash
set -x
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
#
# After https://github.com/equinor/ert/issues/1634 the ERT_SOURCE_ROOT needs to
# point at the ert source root (where .git is).

build_and_test () {
	rm -rf jenkinsbuild
	mkdir jenkinsbuild
	run setup
	run build_libecl
	run build_ert_clib
	run build_ert_dev
	run run_ctest
	run run_pytest_normal
}

setup () {
	run setup_variables
	run create_directories
	run create_virtualenv
	run source_build_tools
	run clone_repos
}

build_libecl () {
	run enable_environment

	pushd $LIBECL_BUILD
	cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALL -DCMAKE_BUILD_TYPE=RelWithDebInfo
	make -j 6 install
	popd
}

build_ert_clib () {
	run enable_environment

	pushd $ERT_CLIB_BUILD
	cmake ${ERT_SOURCE_ROOT}/src/clib \
		  -DCMAKE_PREFIX_PATH=$INSTALL \
		  -DCMAKE_INSTALL_PREFIX=$INSTALL \
		  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
		  -DBUILD_TESTS=ON
	make -j 6 install
	popd
}

build_ert_dev () {
	run enable_environment
	pip install ${ERT_SOURCE_ROOT}
	pip install -r dev-requirements.txt
}

source_build_tools() {
	export PATH=/opt/rh/devtoolset-8/root/bin:$PATH
	python --version
	gcc --version
	cmake --version

	set -e
}

setup_variables () {
	ENV=$WORKING_DIR/venv
	INSTALL=$WORKING_DIR/venv

	LIBECL_ROOT=$WORKING_DIR/libecl
	LIBECL_BUILD=$LIBECL_ROOT/build

	ERT_CLIB_ROOT=$WORKING_DIR/_ert_clib
	ERT_CLIB_BUILD=$ERT_CLIB_ROOT/build
}

enable_environment () {
	run setup_variables
	source $ENV/bin/activate
	run source_build_tools

	export ERT_SHOW_BACKTRACE=Y
	export RMS_SITE_CONFIG=/prog/res/komodo/bleeding-py38-rhel7/root/lib/python3.8/site-packages/ert_configurations/resources/rms_config.yml

	# Conan v1 bundles its own certs due to legacy reasons, so we point it
	# to the system's certs instead.
	export CONAN_CACERT_PATH=/etc/pki/tls/cert.pem
}

create_directories () {
	mkdir $INSTALL
}

clone_repos () {
	echo "Cloning into $LIBECL_ROOT"
	git clone https://github.com/equinor/libecl $LIBECL_ROOT
	mkdir -p $LIBECL_BUILD

	echo "Cloning into $ERT_CLIB_ROOT"
	git clone . $ERT_CLIB_ROOT
	mkdir -p $ERT_CLIB_BUILD
}

create_virtualenv () {
	mkdir $ENV
	source /opt/rh/rh-python38/enable
	python3 -m venv $ENV
	source $ENV/bin/activate
	pip install -U pip wheel setuptools cmake pybind11

	# Conan is a C++ package manager and is required by ecl
	pip install conan
}

run_ctest () {
	run enable_environment
	pushd $ERT_CLIB_BUILD
	export ERT_SITE_CONFIG=${ERT_SOURCE_ROOT}/src/ert/shared/share/ert/site-config
	ctest -j $CTEST_JARG -E Lint --output-on-failure
	popd
}

run_pytest_normal () {
	run enable_environment

	# Avoid implicitly loaded cwd modules
	pushd ${ERT_CLIB_BUILD}
	python -m pytest --durations=10 ${ERT_SOURCE_ROOT}/tests/libres_tests
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
		ERT_SOURCE_ROOT=$(pwd)
	else
		ERT_SOURCE_ROOT=${WORKING_DIR}
fi

if [ $# -ne 0 ]
	then
		run $1
fi
