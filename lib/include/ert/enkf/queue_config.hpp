/*
   Copyright (C) 2017  Equinor ASA, Norway.

   The file 'queue_config.h' is part of ERT - Ensemble based Reservoir Tool.

   ERT is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   ERT is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or
   FITNESS FOR A PARTICULAR PURPOSE.

   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
   for more details.
 */


#ifndef ERT_QUEUE_CONFIG_H
#define ERT_QUEUE_CONFIG_H
#ifdef __cplusplus
extern "C" {
#endif

#include <ert/config/config_parser.hpp>
#include <ert/config/config_content_item.hpp>
#include <ert/config/config_content_node.hpp>
#include <ert/config/config_schema_item.hpp>

#include <ert/job_queue/job_queue.hpp>

#define LSF_DRIVER_NAME    "LSF"
#define LOCAL_DRIVER_NAME  "LOCAL"
#define RSH_DRIVER_NAME    "RSH"
#define TORQUE_DRIVER_NAME "TORQUE"


typedef struct queue_config_struct queue_config_type;

    queue_config_type * queue_config_alloc_load(const char * user_config_file);
    queue_config_type * queue_config_alloc(const config_content_type * config_content);
    queue_config_type * queue_config_alloc_full(char * job_script,
                                                bool user_mode,
                                                int max_submit,
                                                int num_cpu,
                                                int driver_type);
    queue_config_type * queue_config_alloc_local_copy( queue_config_type * queue_config);
    void queue_config_free(queue_config_type * queue_config);

    int queue_config_get_max_submit(queue_config_type * queue_config);
    bool queue_config_has_job_script( const queue_config_type * queue_config );
    bool queue_config_set_job_script(queue_config_type * queue_config, const char * job_script);
    const char * queue_config_get_job_script(const queue_config_type * queue_config);

    job_driver_type queue_config_get_driver_type(const queue_config_type * queue_config);

    queue_driver_type * queue_config_get_queue_driver(const queue_config_type * queue_config,
                                                      const char * driver_name);
    bool queue_config_has_queue_driver(const queue_config_type * queue_config,
                                       const char * driver_name);
    void queue_config_create_queue_drivers(queue_config_type * queue_config);

    /**
     * Queue system is typically one of LSF, LOCAL, TORQUE, RHS, ...  Given a
     * queue system, you can obtain the _driver_ (e.g. lsf_driver).
     *
     * Should not be confused with queue_name, which is typically just a
     * parameter we can send to the LSF cluster to get a certain queue,
     * e.g. "mr".
     */
    const char * queue_config_get_queue_system(const queue_config_type * queue_config);

    void queue_config_add_config_items(config_parser_type * parser, bool site_mode);

    job_queue_type * queue_config_alloc_job_queue(const queue_config_type * queue_config);

    int queue_config_get_num_cpu(const queue_config_type * queue_config);

UTIL_SAFE_CAST_HEADER(queue_config);
UTIL_IS_INSTANCE_HEADER(queue_config);

#ifdef __cplusplus
}
#endif
#endif
