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
	setup
	build_ert_clib
	build_ert_dev
	run_ctest
}

enable_environment () {
	source /opt/rh/devtoolset-8/enable
	source ${ERT_SOURCE_ROOT}/venv/bin/activate

	python --version
	gcc --version
	cmake --version

	export ERT_SHOW_BACKTRACE=Y
	export RMS_SITE_CONFIG=/prog/res/komodo/bleeding-py38-rhel7/root/lib/python3.8/site-packages/ert_configurations/resources/rms_config.yml

	# Conan v1 bundles its own certs due to legacy reasons, so we point it
	# to the system's certs instead.
	export CONAN_CACERT_PATH=/etc/pki/tls/cert.pem
}

setup () {
	/opt/rh/rh-python38/root/usr/bin/python -m venv ${ERT_SOURCE_ROOT}/venv
	${ERT_SOURCE_ROOT}/venv/bin/pip install -U pip wheel setuptools cmake pybind11 "conan<2" ecl
}

build_ert_clib () {
	enable_environment

	mkdir ${ERT_SOURCE_ROOT}/build
	pushd ${ERT_SOURCE_ROOT}/build
	cmake ${ERT_SOURCE_ROOT}/src/clib \
		  -DCMAKE_BUILD_TYPE=Debug
	make -j$(nproc)
	popd
}

build_ert_dev () {
	enable_environment
	pip install ${ERT_SOURCE_ROOT}
	pip install -r ${ERT_SOURCE_ROOT}/dev-requirements.txt
}

run_ctest () {
	enable_environment
	pushd ${ERT_SOURCE_ROOT}/build
	export ERT_SITE_CONFIG=${ERT_SOURCE_ROOT}/src/ert/shared/share/ert/site-config
	ctest -j$(nproc) -E Lint --output-on-failure
	popd
}

ERT_SOURCE_ROOT=$(pwd)

[[ $# != 0 ]] && $1
