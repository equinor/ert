#include <ert/util/util.h>
#include <ert/util/test_util.h>
#include <ert/util/test_work_area.h>

#include <ert/job_queue/job_queue.h>
#include <ert/job_queue/ext_job.h>
#include <ert/job_queue/ext_joblist.h>
#include <ert/job_queue/lsf_driver.h>
#include <ert/job_queue/rsh_driver.h>
#include <ert/job_queue/local_driver.h>
#include <ert/job_queue/queue_driver.h>

#include <ert/config/config_parser.h>

#include <ert/enkf/config_keys.h>
#include <ert/enkf/queue_config.h>


bool some_function(void * param) {
   return true;
}

void test_empty() {
   queue_config_type * queue_config = queue_config_alloc();
   queue_config_free(queue_config);
}


void test_parse() {
   test_work_area_type * work_area = test_work_area_alloc("queue_config");
   config_parser_type * parser = config_alloc( );


   queue_config_add_config_items( parser, true );
   test_assert_true( config_has_schema_item( parser , QUEUE_SYSTEM_KEY ));
   test_assert_true( config_has_schema_item( parser , QUEUE_OPTION_KEY ));
   test_assert_true( config_has_schema_item( parser , JOB_SCRIPT_KEY ));

   {
     FILE* stream1 = util_fopen("tiny_executable", "w");
     fclose(stream1);
     util_chmod_if_owner("tiny_executable", 0777);
   }
   {
     FILE* stream = util_fopen("queue_config.txt", "w");
     fprintf(stream, "QUEUE_SYSTEM LSF\n");
     fprintf(stream, "LSF_SERVER    be-grid01\n");
     fprintf(stream, "QUEUE_OPTION  LSF     BJOBS_CMD   the_path\n");
     fprintf(stream, "JOB_SCRIPT  tiny_executable\n");
     fprintf(stream, "MAX_SUBMIT   6\n");
     fclose(stream);
   }


   config_content_type * config_content = config_parse(parser, "queue_config.txt", NULL, NULL, NULL, NULL, CONFIG_UNRECOGNIZED_ERROR, true);
   test_assert_true(config_content_has_item(config_content, QUEUE_SYSTEM_KEY));

   test_assert_true(config_content_has_item(config_content, QUEUE_OPTION_KEY));
   test_assert_true(config_content_has_item(config_content, MAX_SUBMIT_KEY));

   queue_config_type * queue_config = queue_config_alloc();
   queue_config_init(queue_config, config_content);

   test_assert_true(queue_config_has_queue_driver(queue_config, "LSF"));
   test_assert_true(queue_config_get_driver_type(queue_config) == LSF_DRIVER);
   
   test_check_double_equal(queue_config_get_max_submit(queue_config), 6);

   {
     queue_driver_type * lsf_driver = queue_config_get_queue_driver(queue_config, LSF_DRIVER_NAME);
     test_assert_true(queue_driver_is_instance(lsf_driver));
     test_assert_string_equal(queue_driver_get_option(lsf_driver, LSF_BJOBS_CMD) , "the_path");
   }

   test_assert_true(queue_config_has_job_script(queue_config));
   test_assert_string_equal(queue_config_get_queue_name(queue_config), LSF_DRIVER_NAME);

   //test for licence path
   job_queue_type * job_queue = queue_config_alloc_job_queue(queue_config, some_function, some_function, some_function);
   test_assert_double_equal(job_queue_get_max_submit(job_queue), 6);


 

   job_queue_free(job_queue);
   config_content_free(config_content);
   config_free( parser );
   test_work_area_free( work_area );
}





int main() {
    util_install_signals();
 
    test_empty();
    test_parse();
    return 0;
}
