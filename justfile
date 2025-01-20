# configuration for for `just`

# run poly example test-case
poly:
    ert gui test-data/ert/poly_example/poly.ert

# run snake oil test-case
snake_oil:
    ert gui test-data/ert/snake_oil/snake_oil.ert

# execute rapid unittests
rapid-tests:
    nice pytest -n auto tests/ert/unit_tests --hypothesis-profile=fast -m "not integration_test"
