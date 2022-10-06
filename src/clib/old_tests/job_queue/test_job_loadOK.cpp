#include <stdlib.h>

#include <ert/config/config_parser.hpp>

#include <ert/job_queue/workflow_job.hpp>

bool loadConfig(config_parser_type *config, const char *config_file,
                config_parser_type *config_compiler) {
    bool OK = false;
    workflow_job_type *cmd =
        workflow_job_config_alloc("NAME", config, config_file);

    if (cmd != NULL) {
        OK = true;
        workflow_job_update_config_compiler(cmd, config_compiler);
        workflow_job_free(cmd);
    }

    return OK;
}

int main(int argc, char **argv) {
    int status = 0;
    {
        config_parser_type *config = workflow_job_alloc_config();
        config_parser_type *config_compiler = config_alloc();
        int iarg;
        bool OK = true;

        for (iarg = 1; iarg < argc; iarg++)
            OK = OK && loadConfig(config, argv[iarg], config_compiler);

        if (!OK)
            status = 1;

        config_free(config_compiler);
        config_free(config);
    }
    exit(status);
}
