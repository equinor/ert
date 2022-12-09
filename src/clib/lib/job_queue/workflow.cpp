#include "ert/python.hpp"
#include <filesystem>

#include <stdlib.h>

#include <ert/res_util/subst_list.hpp>
#include <ert/util/int_vector.hpp>
#include <ert/util/stringlist.hpp>
#include <ert/util/vector.hpp>

#include <ert/config/config_parser.hpp>

#include <ert/job_queue/workflow.hpp>

namespace fs = std::filesystem;

#define WORKFLOW_COMMENT_STRING "--"
#define WORKFLOW_INCLUDE "INCLUDE"

typedef struct cmd_struct cmd_type;

struct cmd_struct {
    const workflow_job_type *workflow_job;
    stringlist_type *arglist;
};

struct workflow_struct {
    time_t compile_time;
    bool compiled;
    char *src_file;
    vector_type *cmd_list;
    workflow_joblist_type *joblist;
    std::vector<std::string> last_error;
    vector_type *stack;
};

static cmd_type *cmd_alloc(const workflow_job_type *workflow_job,
                           const stringlist_type *arglist) {
    cmd_type *cmd = (cmd_type *)util_malloc(sizeof *cmd);
    cmd->workflow_job = workflow_job;
    cmd->arglist = stringlist_alloc_deep_copy(arglist);
    return cmd;
}

static void cmd_free(cmd_type *cmd) {
    stringlist_free(cmd->arglist);
    free(cmd);
}

static void cmd_free__(void *arg) {
    auto cmd = static_cast<cmd_type *>(arg);
    cmd_free(cmd);
}

static void workflow_add_cmd(workflow_type *workflow, cmd_type *cmd) {
    vector_append_owned_ref(workflow->cmd_list, cmd, cmd_free__);
}

static void workflow_clear(workflow_type *workflow) {
    vector_clear(workflow->cmd_list);
}

bool workflow_try_compile(workflow_type *script,
                          const subst_list_type *context) {
    if (fs::exists(script->src_file)) {
        const char *src_file = script->src_file;
        char *tmp_file = NULL;
        bool update = false;
        if (context != NULL) {
            tmp_file = util_alloc_tmp_file("/tmp", "ert-workflow", false);
            update =
                subst_list_filter_file(context, script->src_file, tmp_file);
            if (update) {
                script->compiled = false;
                src_file = tmp_file;
            } else {
                remove(tmp_file);
                free(tmp_file);
                tmp_file = NULL;
            }
        }

        {
            time_t src_mtime = util_file_mtime(script->src_file);
            if (script->compiled) {
                if (util_difftime_seconds(src_mtime, script->compile_time) > 0)
                    return true;
                else {
                    // Script has been compiled succesfully, but then changed afterwards.
                    // We try to recompile; if that fails we are left with 'nothing'.
                }
            }
        }

        {
            // Try to compile
            config_parser_type *config_compiler =
                workflow_joblist_get_compiler(script->joblist);
            script->compiled = false;
            workflow_clear(script);
            {
                config_content_type *content =
                    config_parse(config_compiler, src_file,
                                 WORKFLOW_COMMENT_STRING, WORKFLOW_INCLUDE,
                                 NULL, NULL, CONFIG_UNRECOGNIZED_ERROR, true);

                if (config_content_is_valid(content)) {
                    int cmd_line;
                    for (cmd_line = 0;
                         cmd_line < config_content_get_size(content);
                         cmd_line++) {
                        const config_content_node_type *node =
                            config_content_iget_node(content, cmd_line);
                        const char *jobname = config_content_node_get_kw(node);
                        const workflow_job_type *job =
                            workflow_joblist_get_job(script->joblist, jobname);
                        cmd_type *cmd = cmd_alloc(
                            job, config_content_node_get_stringlist(node));

                        workflow_add_cmd(script, cmd);
                    }
                    script->compiled = true;
                } else
                    script->last_error = content->parse_errors;

                config_content_free(content);
            }
        }

        if (tmp_file != NULL) {
            if (script->compiled)
                remove(tmp_file);
            free(tmp_file);
        }
    }

    // It is legal to remove the script after successful compilation but
    // then the context will not be applied at subsequent invocations.
    return script->compiled;
}

bool workflow_run(workflow_type *workflow, void *self, bool verbose,
                  const subst_list_type *context) {
    vector_clear(workflow->stack);
    workflow_try_compile(workflow, context);

    if (workflow->compiled) {
        int icmd;
        for (icmd = 0; icmd < vector_get_size(workflow->cmd_list); icmd++) {
            const cmd_type *cmd =
                (const cmd_type *)vector_iget_const(workflow->cmd_list, icmd);
            void *return_value = workflow_job_run(cmd->workflow_job, self,
                                                  verbose, cmd->arglist);
            vector_push_front_ref(workflow->stack, return_value);
        }
        return true;
    } else
        return false;
}

int workflow_get_stack_size(const workflow_type *workflow) {
    return vector_get_size(workflow->stack);
}

void *workflow_iget_stack_ptr(const workflow_type *workflow, int index) {
    return vector_iget(workflow->stack, index);
}

void *workflow_pop_stack(workflow_type *workflow) {
    return vector_pop_front(workflow->stack);
}

workflow_type *workflow_alloc(const char *src_file,
                              workflow_joblist_type *joblist) {
    auto script = new workflow_type;

    script->src_file = util_alloc_string_copy(src_file);
    script->joblist = joblist;
    script->cmd_list = vector_alloc_new();
    script->compiled = false;
    script->stack = vector_alloc_new();

    workflow_try_compile(script, NULL);
    return script;
}

void workflow_free(workflow_type *workflow) {
    free(workflow->src_file);
    vector_free(workflow->cmd_list);
    vector_free(workflow->stack);

    delete workflow;
}

void workflow_free__(void *arg) {
    auto workflow = static_cast<workflow_type *>(arg);
    workflow_free(workflow);
}

const std::vector<std::string>
workflow_get_last_error(const workflow_type *workflow) {
    return workflow->last_error;
}

int workflow_size(const workflow_type *workflow) {
    return vector_get_size(workflow->cmd_list);
}

const workflow_job_type *workflow_iget_job(const workflow_type *workflow,
                                           int index) {
    const cmd_type *cmd =
        (const cmd_type *)vector_iget_const(workflow->cmd_list, index);
    return cmd->workflow_job;
}

stringlist_type *workflow_iget_arguments(const workflow_type *workflow,
                                         int index) {
    const cmd_type *cmd =
        (const cmd_type *)vector_iget_const(workflow->cmd_list, index);
    return cmd->arglist;
}

extern "C" PY_USED const char *
worflow_get_src_file(const workflow_type *workflow) {
    return workflow->src_file;
}

ERT_CLIB_SUBMODULE("workflow", m) {
    m.def("get_last_error",
          [](Cwrap<workflow_type> self) { return self->last_error; });
}
