#include <stdlib.h>

#include <ert/util/test_util.h>

#include <ert/enkf/active_list.hpp>

int main(int argc, char **argv) {
    ActiveList active_list1;
    ActiveList active_list2;

    test_assert_true(active_list1 == active_list2);
    test_assert_true(active_list1.getMode() == ALL_ACTIVE);

    active_list1.add_index(11);
    test_assert_false(active_list1 == active_list2);
    test_assert_true(active_list1.getMode() == PARTLY_ACTIVE);

    active_list2.add_index(11);
    test_assert_true(active_list1 == active_list2);
}
