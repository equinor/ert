#  Copyright (C) 2015  Equinor ASA, Norway.
#
#  The file 'test_active_list.py' is part of ERT - Ensemble based Reservoir Tool.
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

from res.enkf import ActiveList, ActiveMode


class ActiveListTest(ResTest):
    def test_active_mode_enum(self):
        self.assertEqual(ActiveMode.ALL_ACTIVE, 1)
        self.assertEqual(ActiveMode.PARTLY_ACTIVE, 3)
        self.assertEqual(ActiveMode(1).name, "ALL_ACTIVE")
        self.assertEqual(ActiveMode(3).name, "PARTLY_ACTIVE")

    def test_active_size(self):
        al = ActiveList()
        self.assertEqual(7, al.getActiveSize(7))
        self.assertEqual(-1, al.getActiveSize(-1))

        al.addActiveIndex(10)
        self.assertEqual(1, al.getActiveSize(7))
        al.addActiveIndex(10)
        self.assertEqual(1, al.getActiveSize(7))
        al.addActiveIndex(100)
        self.assertEqual(2, al.getActiveSize(7))

    def test_create(self):
        active_list = ActiveList()
        self.assertEqual(active_list.getMode(), ActiveMode.ALL_ACTIVE)
        active_list.addActiveIndex(10)
        self.assertEqual(active_list.getMode(), ActiveMode.PARTLY_ACTIVE)

    def test_repr(self):
        al = ActiveList()
        rep = repr(al)
        self.assertFalse("PARTLY_ACTIVE" in rep)
        self.assertTrue("ALL_ACTIVE" in rep)
        pfx = "ActiveList("
        self.assertEqual(pfx, rep[: len(pfx)])
        for i in range(150):
            al.addActiveIndex(3 * i)
        rep = repr(al)
        self.assertTrue("150" in rep)
        self.assertTrue("PARTLY_ACTIVE" in rep)
        self.assertFalse("ALL_ACTIVE" in rep)


def test_active_index_list_empty():
    # Empty list
    active_list_obj = ActiveList()
    mode = active_list_obj.getMode()
    assert mode == ActiveMode.ALL_ACTIVE
    list1 = active_list_obj.get_active_index_list()
    assert len(list1) == 0


def test_active_index_list_add_active():
    # add elements, mode is changed to partly active
    active_list_obj = ActiveList()
    assign_list = [0, 1, 4, 12, 88, 77, 5]
    for index in assign_list:
        active_list_obj.addActiveIndex(index)
    mode = active_list_obj.getMode()
    assert mode == ActiveMode.PARTLY_ACTIVE
    list2 = active_list_obj.get_active_index_list()
    # Can not assume that the list is sorted or that it is
    # in the same order as the order the elements are added
    assign_list.sort()
    list2.sort()
    assert assign_list == list2
    default_value = 10
    size = active_list_obj.getActiveSize(default_value)
    assert size == len(list2)


def test_active_index_list_add_more_active():
    active_list_obj = ActiveList()
    assign_list = [0, 1, 4, 12, 88, 77, 5]
    for index in assign_list:
        active_list_obj.addActiveIndex(index)
    # activate more (partly overlapping already activated )
    index = 1
    active_list_obj.addActiveIndex(index)

    list2 = active_list_obj.get_active_index_list()
    list2.sort()

    assign_list2 = [0, 1, 4, 5, 12, 77, 88]
    assert list2 == assign_list2
