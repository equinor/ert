#include <future>

#include <stdlib.h>
#include <unistd.h>

#include <ert/util/test_util.h>
#include <ert/util/test_work_area.h>
#include <ert/util/util.h>
#include <ert/util/vector.h>

#include <ert/ecl/ecl_sum.h>

#include <ert/enkf/enkf_fs.hpp>
#include <ert/enkf/time_map.hpp>

void ecl_test(const char *ecl_case) {
    ecl_sum_type *ecl_sum = ecl_sum_fread_alloc_case(ecl_case, ":");
    time_t start_time = ecl_sum_get_start_time(ecl_sum);
    time_t end_time = ecl_sum_get_end_time(ecl_sum);
    time_map_type *ecl_map = time_map_alloc();

    test_assert_true(time_map_summary_update(ecl_map, ecl_sum).empty());
    test_assert_true(time_map_summary_update(ecl_map, ecl_sum).empty());

    test_assert_time_t_equal(time_map_get_start_time(ecl_map), start_time);
    test_assert_time_t_equal(time_map_get_end_time(ecl_map), end_time);
    test_assert_double_equal(time_map_get_end_days(ecl_map),
                             ecl_sum_get_sim_length(ecl_sum));

    time_map_clear(ecl_map);
    time_map_update(ecl_map, 1, 256);
    test_assert_false(time_map_summary_update(ecl_map, ecl_sum).empty());

    time_map_free(ecl_map);
    ecl_sum_free(ecl_sum);
}

void test_index_map(const char *case1) {
    ecl_sum_type *ecl_sum1 = ecl_sum_fread_alloc_case(case1, ":");

    time_map_type *ecl_map = time_map_alloc();

    {
        int_vector_type *index_map =
            time_map_alloc_index_map(ecl_map, ecl_sum1);
        test_assert_int_equal(int_vector_size(index_map), 0);
        int_vector_free(index_map);
    }

    test_assert_true(time_map_summary_update(ecl_map, ecl_sum1).empty());
    {
        int_vector_type *index_map =
            time_map_alloc_index_map(ecl_map, ecl_sum1);

        for (int i = ecl_sum_get_first_report_step(ecl_sum1);
             i < int_vector_size(index_map); i++)
            test_assert_int_equal(i, int_vector_iget(index_map, i));

        test_assert_int_equal(int_vector_size(index_map),
                              ecl_sum_get_last_report_step(ecl_sum1) + 1);
        int_vector_free(index_map);
    }
}

void simple_test() {
    time_map_type *time_map = time_map_alloc();
    ecl::util::TestArea ta("simple");
    const char *mapfile = "map";

    test_assert_true(time_map_update(time_map, 0, 100));
    test_assert_true(time_map_update(time_map, 1, 200));
    test_assert_true(time_map_update(time_map, 1, 200));

    test_assert_true(time_map_equal(time_map, time_map));
    time_map_fwrite(time_map, mapfile);
    {
        time_map_type *time_map2 = time_map_alloc();

        test_assert_false(time_map_equal(time_map, time_map2));
        time_map_fread(time_map2, mapfile);
        test_assert_true(time_map_equal(time_map, time_map2));
        time_map_free(time_map2);
    }
    {
        time_t mtime1 = util_file_mtime(mapfile);
        sleep(2);
        time_map_fwrite(time_map, mapfile);

        test_assert_time_t_equal(mtime1, util_file_mtime(mapfile));
        time_map_update(time_map, 2, 300);
        time_map_fwrite(time_map, mapfile);
        test_assert_time_t_not_equal(mtime1, util_file_mtime(mapfile));
    }
}

static void simple_update(void *arg) {
    auto tmap = static_cast<time_map_type *>(arg);

    time_map_update(tmap, 0, 101);
}

void simple_test_inconsistent() {
    time_map_type *time_map = time_map_alloc();

    test_assert_true(time_map_update(time_map, 0, 100));

    test_assert_util_abort("time_map_update_abort", simple_update, time_map);

    time_map_free(time_map);
}

#define MAP_SIZE 100

void thread_test() {
    time_map_type *time_map = time_map_alloc();
    test_assert_false(time_map_is_readonly(time_map));

    auto update_time_map = [time_map] {
        int i;
        for (i = 0; i < MAP_SIZE; i++)
            time_map_update(time_map, i, i);

        test_assert_int_equal(MAP_SIZE, time_map_get_size(time_map));
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
    {
        int i;
        for (i = 0; i < MAP_SIZE; i++)
            test_assert_true(time_map_iget(time_map, i) == i);
    }
    time_map_free(time_map);
}

void test_read_only() {
    ecl::util::TestArea ta("read_only");
    {
        time_map_type *tm = time_map_alloc();

        test_assert_false(time_map_is_readonly(tm));

        time_map_update(tm, 0, 0);
        time_map_update(tm, 1, 10);
        time_map_update(tm, 2, 20);

        time_map_fwrite(tm, "case/files/time-map");
        time_map_free(tm);
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

    test_read_only();

    exit(0);
}
