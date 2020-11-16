from ert3.dummy_evaluator import dummy_evaluator

def test_dummy_evaluator():
    coefficients = [(1,2,3)]

    data = dummy_evaluator(coefficients)[0]
    for idx in [0, 9]:
        assert data["polynomial_output"][idx] == 1 * idx ** 2 + 2 * idx + 3
