from ert._c_wrappers.enkf import ActiveList, ActiveMode


def test_active_mode_enum():
    # pylint: disable=no-member
    assert ActiveMode.ALL_ACTIVE == 1
    assert ActiveMode.PARTLY_ACTIVE == 3
    assert ActiveMode(1).name == "ALL_ACTIVE"
    assert ActiveMode(3).name == "PARTLY_ACTIVE"


def test_active_size():
    al = ActiveList()
    assert al.getActiveSize(7) == 7
    assert al.getActiveSize(-1) == -1

    al.addActiveIndex(10)
    assert al.getActiveSize(7) == 1
    al.addActiveIndex(10)
    assert al.getActiveSize(7) == 1
    al.addActiveIndex(100)
    assert al.getActiveSize(7) == 2


def test_create():
    active_list = ActiveList()
    assert active_list.getMode() == ActiveMode.ALL_ACTIVE
    active_list.addActiveIndex(10)
    assert active_list.getMode() == ActiveMode.PARTLY_ACTIVE


def test_repr():
    al = ActiveList()
    rep = repr(al)
    assert "PARTLY_ACTIVE" not in rep
    assert "ALL_ACTIVE" in rep
    pfx = "ActiveList("
    assert pfx == rep[: len(pfx)]
    for i in range(150):
        al.addActiveIndex(3 * i)
    rep = repr(al)
    assert "150" in rep
    assert "PARTLY_ACTIVE" in rep
    assert "ALL_ACTIVE" not in rep


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
