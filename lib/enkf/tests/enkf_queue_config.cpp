#include <ert/util/util.h>
#include <ert/util/test_util.h>
#include <ert/util/test_work_area.hpp>

#include <ert/job_queue/job_queue.hpp>
#include <ert/job_queue/ext_job.hpp>
#include <ert/job_queue/ext_joblist.hpp>
#include <ert/job_queue/lsf_driver.hpp>
#include <ert/job_queue/rsh_driver.hpp>
#include <ert/job_queue/local_driver.hpp>
#include <ert/job_queue/queue_driver.hpp>

#include <ert/config/config_parser.hpp>

#include <ert/enkf/config_keys.hpp>
#include <ert/enkf/queue_config.hpp>


void test_empty() {
   queue_config_type * queue_config = queue_config_alloc_load(NULL);
   queue_config_type * queue_config_copy = queue_config_alloc_local_copy( queue_config );
   queue_config_free(queue_config);
   queue_config_free(queue_config_copy);
}

void test_job_script() {
   ecl::util::TestArea ta("script");
   queue_config_type * queue_config = queue_config_alloc_load(NULL);
   test_assert_false(queue_config_set_job_script( queue_config , "/does/not/exist" ));

   FILE * job_script = util_fopen("Script.sh" , "w");
   fclose( job_script );

   test_assert_false( queue_config_set_job_script( queue_config , "Script.sh" ));

   chmod("Script.sh" , S_IRWXU );
   test_assert_true( queue_config_set_job_script( queue_config , "Script.sh" ));
   test_assert_false( queue_config_set_job_script( queue_config , "DoesNotExits"));

   char * full_path = util_alloc_realpath( "Script.sh" );
   test_assert_string_equal( full_path , queue_config_get_job_script( queue_config));

   free( full_path );
   queue_config_free( queue_config );
}

void test_parse() {
   ecl::util::TestArea ta("parse");
   const char * user_config_file = "queue_config.txt";
   config_parser_type * parser = config_alloc( );

   queue_config_add_config_items( parser, true );
   test_assert_true( config_has_schema_item( parser , QUEUE_SYSTEM_KEY ));
   test_assert_true( config_has_schema_item( parser , QUEUE_OPTION_KEY ));
   test_assert_true( config_has_schema_item( parser , JOB_SCRIPT_KEY ));

   FILE* stream1 = util_fopen("tiny_executable", "w");
   fclose(stream1);
   util_chmod_if_owner("tiny_executable", 0777);

   FILE* stream = util_fopen(user_config_file, "w");
   fprintf(stream, "NUM_REALIZATIONS 14\n");
   fprintf(stream, "QUEUE_SYSTEM LSF\n");
   fprintf(stream, "LSF_SERVER    be-grid01\n");
   fprintf(stream, "QUEUE_OPTION  LSF     BJOBS_CMD   the_path\n");
   fprintf(stream, "JOB_SCRIPT  tiny_executable\n");
   fprintf(stream, "MAX_SUBMIT   6\n");
   fclose(stream);

   config_content_type * config_content = config_parse(parser, user_config_file, NULL, NULL, NULL, NULL, CONFIG_UNRECOGNIZED_ERROR, true);
   test_assert_true(config_content_has_item(config_content, QUEUE_SYSTEM_KEY));

   test_assert_true(config_content_has_item(config_content, QUEUE_OPTION_KEY));
   test_assert_true(config_content_has_item(config_content, MAX_SUBMIT_KEY));

   queue_config_type * queue_config = queue_config_alloc_load(user_config_file);

   test_assert_true(queue_config_has_queue_driver(queue_config, "LSF"));
   test_assert_true(queue_config_get_driver_type(queue_config) == LSF_DRIVER);

   test_check_double_equal(queue_config_get_max_submit(queue_config), 6);

   queue_driver_type * lsf_driver = queue_config_get_queue_driver(queue_config, LSF_DRIVER_NAME);
   test_assert_true(queue_driver_is_instance(lsf_driver));
   test_assert_string_equal((const char *) queue_driver_get_option(lsf_driver, LSF_BJOBS_CMD) , "the_path");

   test_assert_true(queue_config_has_job_script(queue_config));
   test_assert_string_equal(queue_config_get_queue_system(queue_config), LSF_DRIVER_NAME);

   //test for licence path
   job_queue_type * job_queue = queue_config_alloc_job_queue(queue_config);
   test_assert_double_equal(job_queue_get_max_submit(job_queue), 6);


   //testing for copy with local driver only
   queue_config_type * queue_config_copy = queue_config_alloc_local_copy( queue_config );

   test_assert_true( strcmp(queue_config_get_job_script(queue_config),
                            queue_config_get_job_script(queue_config_copy)) == 0);

   test_assert_true( queue_config_get_driver_type(queue_config_copy) == LOCAL_DRIVER );

   test_assert_true( queue_config_get_max_submit(queue_config_copy) == 6);

   queue_config_free( queue_config_copy );
   queue_config_free( queue_config );
   job_queue_free(job_queue);
   config_content_free(config_content);
   config_free( parser );
}

int main() {

    util_install_signals();
    test_empty();
    test_parse();
    test_job_script();
    return 0;
}
