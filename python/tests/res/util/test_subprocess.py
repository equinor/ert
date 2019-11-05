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


import unittest
import os
from tests.utils import tmpdir
from subprocess import Popen, PIPE

from res.util.subprocess import await_process_tee


class TestSubprocess(unittest.TestCase):
    @tmpdir()
    def test_await_process_tee(self):
        with open("a", "wb") as a_fh, open("b", "wb") as b_fh:
            process = Popen(["/bin/cat", "/bin/cat"], stdout=PIPE)
            await_process_tee(process, a_fh, b_fh)

        with open("a", "rb") as f:
            a_content = f.read()
        with open("b", "rb") as f:
            b_content = f.read()
        with open("/bin/cat", "rb") as f:
            cat_content = f.read()

        self.assertTrue(process.stdout.closed)
        self.assertEqual(cat_content, a_content)
        self.assertEqual(cat_content, b_content)
