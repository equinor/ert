import datetime

from ecl.util.util import BoolVector
from ecl.util.test import TestAreaContext
from tests import ResTest
from res.enkf import ObsData, ObsBlock
from res.util import Matrix


class ObsDataTest(ResTest):
    def test_create(self):
        obs_data = ObsData()
        obs_size = 10
        block = obs_data.addBlock("OBS", obs_size)
        self.assertTrue(isinstance(block, ObsBlock))

        block[0] = (100, 10)
        block[1] = (120, 12)
        D = obs_data.createDObs()
        self.assertTrue(isinstance(D, Matrix))
        self.assertEqual(D.dims(), (2, 2))

        self.assertEqual(D[0, 0], 100)
        self.assertEqual(D[1, 0], 120)
        self.assertEqual(D[0, 1], 10)
        self.assertEqual(D[1, 1], 12)

        obs_data.scaleMatrix(D)
        self.assertEqual(D[0, 0], 10)
        self.assertEqual(D[1, 0], 10)
        self.assertEqual(D[0, 1], 1)
        self.assertEqual(D[1, 1], 1)

        R = obs_data.createR()
        self.assertEqual((2, 2), R.dims())

        with self.assertRaises(IndexError):
            obs_data[10]

        v, s = obs_data[0]
        self.assertEqual(v, 100)
        self.assertEqual(s, 10)

        v, s = obs_data[1]
        self.assertEqual(v, 120)
        self.assertEqual(s, 12)

    def test_get_block(self):
        obs_data = ObsData()

        obs_blocks = [("OBS1", 10), ("OBS2", 7), ("OBS3", 15)]
        for name, size in obs_blocks:
            obs_data.addBlock(name, size)

        self.assertEqual(len(obs_blocks), obs_data.get_num_blocks())

        for i in range(obs_data.get_num_blocks()):
            self.assertEqual(obs_blocks[i][0], obs_data.get_block(i).get_obs_key())
