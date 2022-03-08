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

import pytest
from res.enkf.enums import ActiveMode
from res.enkf.local_obsdata import LocalObsdata


def test_empty():
    local_obs_data = LocalObsdata("LOCAL_OBS")
    assert len(local_obs_data) == 0
    assert "NO_SUCH_KEY" not in local_obs_data
    assert "LOCAL_OBS" == local_obs_data.name()


@pytest.mark.parametrize(
    "index, error, expected_msg",
    [
        (10, IndexError, "Invalid index, valid range is"),
        ("NO_SUCH_KEY", KeyError, 'Unknown key "NO_SUCH_KEY'),
    ],
)
def test_exception(index, error, expected_msg):
    local_obs_data = LocalObsdata("LOCAL_OBS")

    with pytest.raises(error, match=expected_msg):
        local_obs_data[index]


@pytest.fixture()
def local_obs_data() -> LocalObsdata:
    local_obs_data = LocalObsdata("LOCAL_OBS")
    local_obs_data.addNode("KEY1")
    local_obs_data.addNode("KEY2")
    local_obs_data.addNode("KEY3")
    yield local_obs_data


def test_add_nodes(local_obs_data):
    assert len(local_obs_data) == 3
    assert "KEY3" in local_obs_data

    for index, key in enumerate(["KEY1", "KEY2", "KEY3"]):
        index_node = local_obs_data[index]
        key_node = local_obs_data[key]

        assert index_node == key_node


def test_del_node(local_obs_data):
    assert len(local_obs_data) == 3
    del local_obs_data["KEY2"]
    assert len(local_obs_data) == 2
    for index, key in enumerate(["KEY1", "KEY3"]):
        index_node = local_obs_data[index]
        key_node = local_obs_data[key]
        assert index_node == key_node


def test_active_list(local_obs_data):
    al1 = local_obs_data.getActiveList("KEY1")
    assert al1.getMode() == ActiveMode.ALL_ACTIVE
    al1.addActiveIndex(1)
    al1.addActiveIndex(3)
    assert al1.getActiveSize(0) == 2

    al2 = local_obs_data.getActiveList("KEY1")
    assert al2.getMode() == ActiveMode.PARTLY_ACTIVE
    assert al2.getActiveSize(0) == 2
