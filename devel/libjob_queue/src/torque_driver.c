/*
   Copyright (C) 2013  Statoil ASA, Norway. 
    
   The file 'torque_driver.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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
#include <string.h>
#include <ert/util/util.h>
#include <ert/util/type_macros.h>
#include <ert/job_queue/torque_driver.h>


#define TORQUE_DRIVER_TYPE_ID 34873653
#define DEFAULT_QSUB_CMD   "qsub"
#define DEFAULT_QSTAT_CMD  "qstat"
#define DEFAULT_QDEL_CMD  "qdel"

struct torque_driver_struct {
  UTIL_TYPE_ID_DECLARATION;
  char * qsub_cmd;
  char * qstat_cmd;
  char * qdel_cmd;
};

UTIL_SAFE_CAST_FUNCTION(torque_driver, TORQUE_DRIVER_TYPE_ID);

static UTIL_SAFE_CAST_FUNCTION_CONST(torque_driver, TORQUE_DRIVER_TYPE_ID)

static void torque_driver_set_qsub_cmd(torque_driver_type * driver, const char * qsub_cmd) {
  driver->qsub_cmd = util_realloc_string_copy(driver->qsub_cmd, qsub_cmd);
}

static void torque_driver_set_qstat_cmd(torque_driver_type * driver, const char * qstat_cmd) {
  driver->qstat_cmd = util_realloc_string_copy(driver->qstat_cmd, qstat_cmd);
}

static void torque_driver_set_qdel_cmd(torque_driver_type * driver, const char * qstat_cmd) {
  driver->qdel_cmd = util_realloc_string_copy(driver->qdel_cmd, qstat_cmd);
}

bool torque_driver_set_option(void * __driver, const char * option_key, const void * value) {
  torque_driver_type * driver = torque_driver_safe_cast(__driver);
  bool has_option = true;
  {
    if (strcmp(TORQUE_QSUB_CMD, option_key) == 0)
      torque_driver_set_qsub_cmd(driver, value);
    else if (strcmp(TORQUE_QSTAT_CMD, option_key) == 0)
      torque_driver_set_qstat_cmd(driver, value);
    else if (strcmp(TORQUE_QDEL_CMD, option_key) == 0)
      torque_driver_set_qdel_cmd(driver, value);
    else
      has_option = false;
  }
  return has_option;
}

const void * torque_driver_get_option(const void * __driver, const char * option_key) {
  const torque_driver_type * driver = torque_driver_safe_cast_const(__driver);
  {
    if (strcmp(TORQUE_QSUB_CMD, option_key) == 0)
      return driver->qsub_cmd;
    else if (strcmp(TORQUE_QSTAT_CMD, option_key) == 0)
      return driver->qstat_cmd;
    else if (strcmp(TORQUE_QDEL_CMD, option_key) == 0)
      return driver->qdel_cmd;
    else {
      util_abort("%s: option_id:%s not recognized for TORQUE driver \n", __func__, option_key);
      return NULL;
    }
  }
}

void * torque_driver_alloc() {
  torque_driver_type * torque_driver = util_malloc(sizeof * torque_driver);
  UTIL_TYPE_ID_INIT(torque_driver, TORQUE_DRIVER_TYPE_ID);
  
  torque_driver->qsub_cmd = NULL;
  torque_driver->qstat_cmd = NULL;
  torque_driver->qdel_cmd = NULL;
  
  torque_driver_set_option(torque_driver, TORQUE_QSUB_CMD, DEFAULT_QSUB_CMD);
  torque_driver_set_option(torque_driver, TORQUE_QSTAT_CMD, DEFAULT_QSTAT_CMD);
  torque_driver_set_option(torque_driver, TORQUE_QDEL_CMD, DEFAULT_QDEL_CMD);
  return torque_driver;
}
