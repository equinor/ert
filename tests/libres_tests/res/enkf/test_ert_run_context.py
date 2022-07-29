#  Copyright (C) 2018  Equinor ASA, Norway.
#
#  The file 'test_ert_run_context.py' is part of ERT - Ensemble based Reservoir Tool.
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


from ecl.util.test import TestAreaContext
from ...libres_utils import ResTest, tmpdir

from res.enkf import ErtRunContext
from res.enkf.runpaths import Runpaths


class ErtRunContextTest(ResTest):
    def test_case_init(self):
        mask = [True] * 100
        ErtRunContext(mask=mask)

    @tmpdir()
    def test_create(self):
        with TestAreaContext("run_context"):

            mask = [True] * 100
            mask[50] = False
            runpaths = Runpaths(
                runpath_format="path/to/sim%d",
                job_name_format="job%d",
            )
            itr = 0
            realizations = list(range(len(mask)))
            run_context1 = ErtRunContext(
                None,
                None,
                mask,
                runpaths.get_paths(realizations, itr),
                runpaths.get_jobnames(realizations, itr),
                itr,
            )
            run_id1 = run_context1.get_id()

            run_arg0 = run_context1[0]
            with self.assertRaises(ValueError):
                run_arg0.getQueueIndex()

            self.assertEqual(run_arg0.iter_id, itr)
            self.assertEqual(run_id1, run_arg0.get_run_id())

            run_context2 = ErtRunContext(
                None,
                None,
                mask,
                runpaths.get_paths(realizations, itr),
                runpaths.get_jobnames(realizations, itr),
                itr,
            )
            run_id2 = run_context2.get_id()

            assert run_id1 != run_id2

            self.assertTrue(run_context1.is_active(49))
            self.assertFalse(run_context1.is_active(50))
