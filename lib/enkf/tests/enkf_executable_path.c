#include <ert/util/util.h>
#include <ert/util/test_util.h>
#include <ert/util/test_work_area.h>

#include <ert/enkf/config_keys.h>
#include <ert/enkf/queue_config.h>

int main() {
    util_install_signals();

    test_work_area_type * work_area = test_work_area_alloc("enkf_executable_path");
    const char * user_config_file = "path.txt";

    config_parser_type * parser = config_alloc( );
    queue_config_add_config_items( parser, true );
    test_assert_true( config_has_schema_item( parser , JOB_SCRIPT_KEY ) );

    FILE* stream = util_fopen(user_config_file, "w");
    fprintf(stream, "NUM_REALIZATIONS 14\n");
    fprintf(stream, "JOB_SCRIPT  ls\n");
    fclose(stream);

    queue_config_alloc_load( user_config_file );

    test_work_area_free( work_area );

    return 0;
}
