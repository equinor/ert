#include <ert/util/util.h>
#include <ert/util/test_util.h>
#include <ert/util/test_work_area.hpp>

#include <ert/enkf/config_keys.hpp>
#include <ert/enkf/queue_config.hpp>

int main() {
    util_install_signals();
    ecl::util::TestArea ta("executable");
    const char * user_config_file = "path.txt";

    config_parser_type * parser = config_alloc( );
    queue_config_add_config_items( parser, true );
    test_assert_true( config_has_schema_item( parser , JOB_SCRIPT_KEY ) );

    FILE* stream = util_fopen(user_config_file, "w");
    fprintf(stream, "NUM_REALIZATIONS 14\n");
    fprintf(stream, "JOB_SCRIPT  ls\n");
    fclose(stream);

    queue_config_alloc_load( user_config_file );
    return 0;
}
