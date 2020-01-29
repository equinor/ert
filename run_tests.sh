run_ert_tests () {
    export TRAVIS_PYTHON_VERSION=$1
    conda deactivate
    conda env remove --name ert_env

    if [[ "$TRAVIS_PYTHON_VERSION" == "2.7" ]]; then
        conda config --set restore_free_channel true;        
        conda create -n ert_env pyqt=4.11.3 python=2.7 --yes;
    else
        conda create -n ert_env python=$TRAVIS_PYTHON_VERSION --yes;
    fi

    conda activate ert_env
    conda install virtualenv --yes
    export LD_LIBRARY_PATH="/data/work/install/ert_37/lib64"
    export PYTHONPATH="/data/work/install/ert_37/lib/python3.7/site-packages"
    export PATH=$CONDA_PREFIX/bin/python:$PATH
    pip install tox
    tox -e $(echo py$TRAVIS_PYTHON_VERSION | tr -d .)
}
