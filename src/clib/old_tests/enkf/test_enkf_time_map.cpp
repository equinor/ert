#include <future>

#include <stdlib.h>
#include <unistd.h>

#include <ert/util/test_util.h>
#include <ert/util/test_work_area.h>
#include <ert/util/util.h>

#include <ert/ecl/ecl_sum.h>

#include <ert/enkf/enkf_fs.hpp>
#include <ert/enkf/time_map.hpp>

void ecl_test(const char *ecl_case) {
    ecl_sum_type *ecl_sum = ecl_sum_fread_alloc_case(ecl_case, ":");
    time_t start_time = ecl_sum_get_start_time(ecl_sum);
    time_t end_time = ecl_sum_get_end_time(ecl_sum);
    TimeMap time_map;

    test_assert_true(time_map.summary_update(ecl_sum).empty());
    test_assert_true(time_map.summary_update(ecl_sum).empty());

    test_assert_time_t_equal(time_map.start_time(), start_time);
    test_assert_time_t_equal(time_map.end_time(), end_time);
    test_assert_double_equal(time_map.end_days(),
                             ecl_sum_get_sim_length(ecl_sum));

    time_map.clear();
    time_map.update(1, 256);
    test_assert_false(time_map.summary_update(ecl_sum).empty());

    ecl_sum_free(ecl_sum);
}

void test_index_map(const char *case1) {
    ecl_sum_type *ecl_sum1 = ecl_sum_fread_alloc_case(case1, ":");

    TimeMap time_map;

    {
        std::vector<int> index_map = time_map.make_index_map(ecl_sum1);
        test_assert_int_equal(index_map.size(), 0);
    }

    test_assert_true(time_map.summary_update(ecl_sum1).empty());
    {
        auto index_map = time_map.make_index_map(ecl_sum1);

        for (int i = ecl_sum_get_first_report_step(ecl_sum1);
             i < index_map.size(); i++)
            test_assert_int_equal(i, index_map[i]);

        test_assert_int_equal(index_map.size(),
                              ecl_sum_get_last_report_step(ecl_sum1) + 1);
    }
}

void simple_test() {
    TimeMap time_map;
    ecl::util::TestArea ta("simple");
    const char *mapfile = "map";

    test_assert_true(time_map.try_update(0, 100));
    test_assert_true(time_map.try_update(1, 200));
    test_assert_true(time_map.try_update(1, 200));

    test_assert_true(time_map == time_map);
    time_map.write(mapfile);
    {
        TimeMap time_map2;

        test_assert_false(time_map == time_map2);
        time_map2.read_binary(mapfile);
        test_assert_true(time_map == time_map2);
    }
}

void simple_test_inconsistent() {
    TimeMap time_map;

    test_assert_true(time_map.try_update(0, 100));

    bool exception_thrown{};
    try {
        time_map.update(0, 101);
    } catch (std::runtime_error &) {
        exception_thrown = true;
    }
    test_assert_true(exception_thrown);
}

#define MAP_SIZE 100

void thread_test() {
    TimeMap time_map;

    auto update_time_map = [&time_map] {
        int i;
        for (i = 0; i < MAP_SIZE; i++)
            time_map.update(i, i);

        test_assert_int_equal(MAP_SIZE, time_map.size());
    };

    {
        std::vector<std::future<void>> futures;
        int pool_size = 50;
        for (int i{}; i < pool_size; ++i)
            futures.emplace_back(
                std::async(std::launch::async, update_time_map));

        for (auto &future : futures)
            future.get();
    }
}

int main(int argc, char **argv) {

    util_install_signals();

    if (argc == 1) {
        simple_test();
        simple_test_inconsistent();
        thread_test();
    } else {
        ecl_test(argv[1]);
        test_index_map(argv[1]);
    }

    exit(0);
}
