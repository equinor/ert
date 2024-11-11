# configuration for for `just`

# run poly example test-case
poly:
    ert gui test-data/ert/poly_example/poly.ert

# run snake oil test-case
snake_oil:
    ert gui test-data/ert/snake_oil/snake_oil.ert

# execute rapid unittests
rapid-tests:
    pytest -n logical tests/ert/unit_tests -m "not integration_tests"
