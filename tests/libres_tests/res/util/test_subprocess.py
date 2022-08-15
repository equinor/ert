#  Copyright (C) 2019  Equinor ASA, Norway.
#
#  The file 'test_subprocess.py' is part of ERT - Ensemble based Reservoir Tool.
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
import unittest
from subprocess import PIPE, Popen, TimeoutExpired

from ert._c_wrappers.util.subprocess import await_process_tee

from ...libres_utils import tmpdir


def _find_system_pipe_max_size():
    """This method finds the limit of the system pipe-buffer which
    might be taken into account when using subprocesses with pipes."""
    p = Popen(["dd", "if=/dev/zero", "bs=1"], stdin=PIPE, stdout=PIPE)
    try:
        p.wait(timeout=1)
    except TimeoutExpired:
        p.kill()
        return len(p.stdout.read())


_maxBytes = _find_system_pipe_max_size() - 1


class TestSubprocess(unittest.TestCase):
    @tmpdir()
    def test_await_process_tee(self):
        with open("original", "wb") as fh:
            fh.write(bytearray(os.urandom(_maxBytes)))

        with open("a", "wb") as a_fh, open("b", "wb") as b_fh:
            process = Popen(["/bin/cat", "original"], stdout=PIPE)
            await_process_tee(process, a_fh, b_fh)

        with open("a", "rb") as f:
            a_content = f.read()
        with open("b", "rb") as f:
            b_content = f.read()
        with open("original", "rb") as f:
            original_content = f.read()

        self.assertTrue(process.stdout.closed)
        self.assertEqual(original_content, a_content)
        self.assertEqual(original_content, b_content)

    @tmpdir()
    def test_await_process_finished_tee(self):
        with open("original", "wb") as fh:
            fh.write(bytearray(os.urandom(_maxBytes)))

        with open("a", "wb") as a_fh, open("b", "wb") as b_fh:
            process = Popen(["/bin/cat", "original"], stdout=PIPE)
            process.wait()
            await_process_tee(process, a_fh, b_fh)

        with open("a", "rb") as f:
            a_content = f.read()
        with open("b", "rb") as f:
            b_content = f.read()
        with open("original", "rb") as f:
            original_content = f.read()

        self.assertTrue(process.stdout.closed)
        self.assertEqual(original_content, a_content)
        self.assertEqual(original_content, b_content)
