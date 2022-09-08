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


def test_all_active(snake_oil_field_example):
    """ERT by default updates all parameters and observations
    as defined in the config file.
    """
    update_step = snake_oil_field_example.update_configuration[0]
    assert [param.name for param in update_step.parameters] == [
        "PERMX",
        "PORO",
        "SNAKE_OIL_PARAM",
    ]

    assert len(update_step.observations) == 8
