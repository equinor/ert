#include <stdlib.h>

#include <ert/util/test_util.hpp>
#include <ert/util/test_work_area.hpp>

#include <ert/job_queue/lsf_driver.hpp>

void test_empty_file() {
    const char *stdout_file = "bsub_empty";
    {
        FILE *stream = util_fopen(stdout_file, "w");
        fclose(stream);
    }
    test_assert_int_equal(lsf_job_parse_bsub_stdout("bsub", stdout_file), 0);
}

void test_OK() {
    const char *stdout_file = "bsub_OK";
    {
        FILE *stream = util_fopen(stdout_file, "w");
        fprintf(stream,
                "Job <12345> is submitted to default queue <normal>.\n");
        fclose(stream);
    }
    test_assert_int_equal(lsf_job_parse_bsub_stdout("bsub", stdout_file),
                          12345);
}

void test_file_does_not_exist() {
    test_assert_int_equal(lsf_job_parse_bsub_stdout("bsub", "does/not/exist"),
                          0);
}

void parse_invalid(void *arg) {
    const char *filename = (const char *)arg;
    lsf_job_parse_bsub_stdout("bsub", filename);
}

void test_parse_fail_abort() {
    const char *stdout_file = "bsub_abort";
    {
        FILE *stream = util_fopen(stdout_file, "w");
        fprintf(stream, "Job 12345 is submitted to default queue <normal>.\n");
        fclose(stream);
    }
    test_assert_util_abort("lsf_job_parse_bsub_stdout", parse_invalid,
                           (void *)stdout_file);
}

int main(int argc, char **argv) {
    ecl::util::TestArea ta("lsf_parse");
    {
        test_empty_file();
        test_file_does_not_exist();
        test_OK();
        test_parse_fail_abort();
    }
}
