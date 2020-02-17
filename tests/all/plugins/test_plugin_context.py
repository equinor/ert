# -*- coding: utf-8 -*-
import os
import sys
import unittest
from _pytest.monkeypatch import MonkeyPatch

from ert_shared.plugins import ErtPluginContext
import tests.all.plugins.dummy_plugins as dummy_plugins


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
        pass

    def tearDown(self):
        self.monkeypatch.undo()

    @unittest.skipIf(sys.version_info.major < 3, "Plugin Manager is Python 3 only")
    def test_no_plugins(self):
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

    @unittest.skipIf(sys.version_info.major < 3, "Plugin Manager is Python 3 only")
    def test_with_plugins(self):
        self.monkeypatch.delenv("ERT_SITE_CONFIG", raising=False)
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

    @unittest.skipIf(sys.version_info.major < 3, "Plugin Manager is Python 3 only")
    def test_already_set(self):
        for var in env_vars:
            self.monkeypatch.setenv(var, "TEST")

        with ErtPluginContext(plugins=[dummy_plugins]) as c:
            for var in env_vars:
                self.assertEqual("TEST", os.environ[var])

        for var in env_vars:
            self.assertEqual("TEST", os.environ[var])

    @unittest.skipIf(
        sys.version_info.major > 2, "Skipping Plugin Manager Python 2 test"
    )
    def test_plugin_context_python_2(self):
        with ErtPluginContext(plugins=[]):
            for var in env_vars:
                with self.assertRaises(KeyError):
                    os.environ[var]
