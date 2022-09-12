#  Copyright (C) 2018  Equinor ASA, Norway.
#
#  The file 'test_rms_config.py' is part of ERT - Ensemble based Reservoir Tool.
#
#  ERT is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
#  for more details.
import os
import stat

import pytest

from ert._c_wrappers.fm.rms import RMSConfig


@pytest.mark.usefixtures("use_tmpdir")
def test_load(monkeypatch):
    monkeypatch.setenv("RMS_SITE_CONFIG", "file/does/not/exist")
    with pytest.raises(IOError):
        conf = RMSConfig()

    monkeypatch.setenv("RMS_SITE_CONFIG", RMSConfig.DEFAULT_CONFIG_FILE)
    conf = RMSConfig()

    with pytest.raises(OSError):
        # pylint: disable=pointless-statement
        conf.executable

    with open("file.yml", "w") as f:
        f.write("this:\n -should\n-be\ninvalid:yaml?")

    monkeypatch.setenv("RMS_SITE_CONFIG", "file.yml")
    with pytest.raises(ValueError):
        conf = RMSConfig()

    os.mkdir("bin")
    with open("bin/rms", "w") as f:
        f.write("This is an RMS executable ...")
    os.chmod("bin/rms", stat.S_IEXEC)

    with open("file.yml", "w") as f:
        f.write("executable: bin/rms")

    conf = RMSConfig()
    assert conf.executable == "bin/rms"
    assert conf.threads is None

    with open("file.yml", "w") as f:
        f.write("executable: bin/rms\n")
        f.write("threads: 17")

    conf = RMSConfig()
    assert conf.threads == 17

    with open("file.yml", "w") as f:
        f.write("executable: bin/rms\n")
        f.write("wrapper: not-exisiting-exec")

    conf = RMSConfig()

    with pytest.raises(OSError):
        # pylint: disable=pointless-statement
        conf.wrapper

    with open("file.yml", "w") as f:
        f.write("executable: bin/rms\n")
        f.write("wrapper: bash")

    conf = RMSConfig()
    assert conf.wrapper == "bash"


@pytest.mark.usefixtures("use_tmpdir")
def test_load_env(monkeypatch):
    monkeypatch.setenv("RMS_SITE_CONFIG", "file.yml")
    with open("file.yml", "w") as f:
        f.write(
            """\
executable: bin/rms\n
wrapper: bash
env:
    10.1.3:
        PATH_PREFIX: /some/path
        PYTHONPATH: /some/pythonpath
"""
        )
    conf = RMSConfig()
    assert conf.env("10.1.3")["PATH_PREFIX"] == "/some/path"
    assert conf.env("10.1.3")["PYTHONPATH"] == "/some/pythonpath"
    assert conf.env("non_existing") == {}
