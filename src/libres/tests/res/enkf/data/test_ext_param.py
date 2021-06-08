import os.path
import json

from res.enkf.data import ExtParam
from res.enkf.config import ExtParamConfig
from ecl.util.test import TestAreaContext
from tests import ResTest


class ExtParamTest(ResTest):
    def test_config(self):
        input_keys = ["key1", "key2", "key3"]
        config = ExtParamConfig("Key", input_keys)
        self.assertTrue(len(config), 3)

        for i in range(len(config)):
            configkey, _ = config[i]
            self.assertEqual(configkey, input_keys[i])

        with self.assertRaises(IndexError):
            c = config[100]

        keys = []
        for key in config.keys():
            keys.append(key)
        self.assertEqual(keys, input_keys)

        self.assertIn("key1", config)

    def test_config_with_suffixes(self):
        input_suffixes = [
            ["a", "b", "c"],
            ["2"],
            ["asd", "qwe", "zxc"],
        ]
        input_dict = {
            "key1": input_suffixes[0],
            "key2": input_suffixes[1],
            "key3": input_suffixes[2],
        }
        config = ExtParamConfig("Key", input_dict)

        self.assertTrue(len(config), 3)
        self.assertIn("key3", config)
        self.assertNotIn("not_me", config)
        self.assertIn(("key3", "asd"), config)
        self.assertNotIn(("key3", "not_me_either"), config)
        self.assertNotIn(("who", "b"), config)

        for i in range(len(config)):
            configkey, configsuffixes = config[i]
            self.assertIn(configkey, input_dict)
            self.assertIn(configsuffixes, input_suffixes)

        for k in input_dict:
            configsuffixes = config[k]
            self.assertIn(configsuffixes, input_suffixes)

        with self.assertRaises(IndexError):
            c = config[100]

        with self.assertRaises(IndexError):
            c = config["no_such_key"]

        self.assertEqual(set(config.keys()), set(input_dict.keys()))

        d = {k: s for k, s in config.items()}
        self.assertEqual(d, input_dict)

    def test_data(self):
        input_keys = ["key1", "key2", "key3"]
        config = ExtParamConfig("Key", input_keys)
        data = ExtParam(config)

        with self.assertRaises(IndexError):
            d = data[100]
        with self.assertRaises(IndexError):
            d = data[-4]

        with self.assertRaises(KeyError):
            d = data["NoSuchKey"]
        with self.assertRaises(KeyError):
            d = data["key1", "a_suffix"]

        self.assertIn("key1", data)
        data[0] = 177
        self.assertEqual(data[0], 177)

        data["key2"] = 321
        self.assertEqual(data[-2], 321)

        with self.assertRaises(ValueError):
            data.set_vector([1, 2])

        data.set_vector([1, 2, 3])
        for i in range(len(data)):
            self.assertEqual(i + 1, data[i])

        with TestAreaContext("json"):
            data.export("file.json")
            d = json.load(open("file.json"))
        for key in data.config.keys():
            self.assertEqual(data[key], d[key])

    def test_data_with_suffixes(self):
        input_suffixes = [
            ["a", "b", "c"],
            ["2"],
            ["asd", "qwe", "zxc"],
        ]
        input_dict = {
            "key1": input_suffixes[0],
            "key2": input_suffixes[1],
            "key3": input_suffixes[2],
        }
        config = ExtParamConfig("Key", input_dict)
        data = ExtParam(config)

        with self.assertRaises(IndexError):
            d = data[0]  # Cannot use indices when we have suffixes
        with self.assertRaises(TypeError):
            d = data["key1", 1]
        with self.assertRaises(KeyError):
            d = data["NoSuchKey"]
        with self.assertRaises(KeyError):
            d = data["key1"]  # requires a suffix
        with self.assertRaises(KeyError):
            d = data["key1", "no_such_suffix"]

        data["key1", "a"] = 1
        data["key1", "b"] = 500.5
        data["key2", "2"] = 2.1
        data["key3", "asd"] = -85
        self.assertEqual(data["key1", "a"], 1)
        self.assertEqual(data["key1", "b"], 500.5)
        self.assertEqual(data["key2", "2"], 2.1)
        self.assertEqual(data["key3", "asd"], -85)

        # We don't know what the value is, but it should be possible to read it
        _ = data["key3", "zxc"]
