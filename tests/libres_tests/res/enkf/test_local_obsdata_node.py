from res.enkf import LocalObsdataNode


def test_tstep():
    node = LocalObsdataNode("KEY")
    assert node.allTimeStepActive()
    assert not node.tstepActive(10)
    assert not node.tstepActive(0)

    node.addTimeStep(10)
    assert not node.allTimeStepActive()

    assert node.tstepActive(10)
    assert not node.tstepActive(0)
