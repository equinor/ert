# -*- coding: utf-8 -*-
import os
import tempfile
import unittest
from unittest.mock import Mock

import tests.ert_tests.all.plugins.dummy_plugins as dummy_plugins
from ert.shared.plugins import ErtPluginContext
from pytest import MonkeyPatch

env_vars = [
    "ECL100_SITE_CONFIG",
    "ECL300_SITE_CONFIG",
    "FLOW_SITE_CONFIG",
    "RMS_SITE_CONFIG",
    "ERT_SITE_CONFIG",
]


class PluginContextTest(unittest.TestCase):
    def setUp(self):
        self.monkeypatch = MonkeyPatch()

    def tearDown(self):
        self.monkeypatch.undo()

    def test_no_plugins(self):
        # pylint: disable=pointless-statement
        self.monkeypatch.delenv("ERT_SITE_CONFIG", raising=False)
        with ErtPluginContext(plugins=[]) as c:
            with self.assertRaises(KeyError):
                os.environ["ECL100_SITE_CONFIG"]
            with self.assertRaises(KeyError):
                os.environ["ECL300_SITE_CONFIG"]
            with self.assertRaises(KeyError):
                os.environ["FLOW_SITE_CONFIG"]
            with self.assertRaises(KeyError):
                os.environ["RMS_SITE_CONFIG"]

            self.assertTrue(os.path.isfile(os.environ["ERT_SITE_CONFIG"]))
            with open(os.environ["ERT_SITE_CONFIG"]) as f:
                self.assertEqual(f.read(), c.plugin_manager.get_site_config_content())

            path = os.environ["ERT_SITE_CONFIG"]

        with self.assertRaises(KeyError):
            os.environ["ERT_SITE_CONFIG"]
        self.assertFalse(os.path.isfile(path))

    def test_with_plugins(self):
        # pylint: disable=pointless-statement
        self.monkeypatch.delenv("ERT_SITE_CONFIG", raising=False)
        # We are comparing two function calls, both of which generate a tmpdir,
        # this makes sure that the same tmpdir is called on both occasions.
        self.monkeypatch.setattr(
            tempfile, "mkdtemp", Mock(return_value=tempfile.mkdtemp())
        )
        with ErtPluginContext(plugins=[dummy_plugins]) as c:
            with self.assertRaises(KeyError):
                os.environ["ECL100_SITE_CONFIG"]
            with self.assertRaises(KeyError):
                os.environ["ECL300_SITE_CONFIG"]
            with self.assertRaises(KeyError):
                os.environ["FLOW_SITE_CONFIG"]
            with self.assertRaises(KeyError):
                os.environ["RMS_SITE_CONFIG"]

            self.assertTrue(os.path.isfile(os.environ["ERT_SITE_CONFIG"]))
            with open(os.environ["ERT_SITE_CONFIG"]) as f:
                self.assertEqual(f.read(), c.plugin_manager.get_site_config_content())

            path = os.environ["ERT_SITE_CONFIG"]

        with self.assertRaises(KeyError):
            os.environ["ERT_SITE_CONFIG"]
        self.assertFalse(os.path.isfile(path))

    def test_already_set(self):
        for var in env_vars:
            self.monkeypatch.setenv(var, "TEST")

        with ErtPluginContext(plugins=[dummy_plugins]):
            for var in env_vars:
                self.assertEqual("TEST", os.environ[var])

        for var in env_vars:
            self.assertEqual("TEST", os.environ[var])
