#! /bin/bash
mypy --config-file=.mypy-strict.ini src/ert/config && python -m pytest -x -n 4 --assert=plain tests/unit_tests/config/
if [ $? -ne 0 ]
then
    exit 1
else
    exit 0
fi
