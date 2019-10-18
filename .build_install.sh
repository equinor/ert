#!bin/bash

function build_install() {
  pushd $1
  pip install -r requirements.txt
  mkdir build
  pushd build
  cmake .. -DENABLE_PYTHON=ON \
-DBUILD_APPLICATIONS=ON \
-DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
-DCMAKE_PREFIX_PATH=$INSTALL_DIR \
-DCMAKE_INSTALL_NAME_DIR=$INSTALL_DIR/lib 
  make install
  popd
  popd
}

build_install $1

