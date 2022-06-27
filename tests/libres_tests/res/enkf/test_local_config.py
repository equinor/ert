#  Copyright (C) 2017  Equinor ASA, Norway.
#
#  This file is part of ERT - Ensemble based Reservoir Tool.
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


from libres_utils import ResTest

from res.test import ErtTestContext


class LocalConfigTest(ResTest):
    def setUp(self):
        self.config = self.createTestPath("local/snake_oil_field/snake_oil.ert")

    def test_all_active(self):
        with ErtTestContext(self.config) as test_context:
            main = test_context.getErt()

            updatestep = main.update_configuration
            update_step = updatestep[0]
            self.assertEqual(3, len(update_step.parameters))
            self.assertEqual(
                ["PERMX", "PORO", "SNAKE_OIL_PARAM"],
                [param.name for param in update_step.parameters],
            )

            self.assertEqual(len(update_step.observations), 8)
