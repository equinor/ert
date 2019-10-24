import collections
import json
import os
import unittest


import ert3


from tests.utils import tmpdir


class TestObservations(unittest.TestCase):

    @tmpdir()
    def test_register_as_resource(self):
        Observation = collections.namedtuple("Observation", ["name", "type", "observations"])
        SingleObs = collections.namedtuple("SingleObs", ["value", "std"])

        observation = Observation(
            name="my_obs",
            type="array",
            observations=(SingleObs(value=val, std=val/10.) for val in range(5)),
        )

        with open("register.json", "w") as f:
            json.dump({}, f)
        os.mkdir("storage")

        ert3.observations.register_as_resource(observation, "register.json", "storage")
        
        with open("register.json") as f:
            register = json.load(f)

        self.assertIn("my_obs", register)
        self.assertTrue(os.path.isfile(register["my_obs"]))
