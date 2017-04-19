#ifndef ERT_QUEUE_CONFIG_H
#define ERT_QUEUE_CONFIG_H
#ifdef __cplusplus
extern "C" {
#endif

#include <ert/config/config_parser.h>
#include <ert/config/config_content_item.h>
#include <ert/config/config_content_node.h>
#include <ert/config/config_schema_item.h>

#define LSF_DRIVER_NAME    "LSF"
#define LOCAL_DRIVER_NAME  "LOCAL"
#define RSH_DRIVER_NAME    "RSH"
#define TORQUE_DRIVER_NAME "TORQUE"


typedef struct queue_config_struct queue_config_type;

    queue_config_type * queue_config_alloc();
    void queue_config_free(queue_config_type * queue_config);
    void queue_config_init_user_mode(queue_config_type * queue_config);
    bool queue_config_init(queue_config_type * queue_config, const config_content_type * config);

    bool queue_config_has_job_script( const queue_config_type * queue_config );
    const char * queue_config_get_job_script(const queue_config_type * queue_config);
    bool queue_config_set_job_script(queue_config_type * queue_config, const char * job_script);

    job_driver_type queue_config_get_driver_type(const queue_config_type * queue_config);
    queue_driver_type * queue_config_get_queue_driver(const queue_config_type * queue_config, const char * driver_name);
    bool queue_config_has_queue_driver(const queue_config_type * queue_config, const char * driver_name);
    void queue_config_create_queue_drivers(queue_config_type * queue_config);
    const char * queue_config_get_queue_name(const queue_config_type * queue_config);

    void queue_config_add_config_items(config_parser_type * parser, bool site_mode);


#ifdef __cplusplus
}
#endif
#endif

