/*
   Copyright (C) 2013  Statoil ASA, Norway. 
    
   The file 'enkf_ecl_config.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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
#include <ert/util/test_util.h>
#include <ert/enkf/qc_module.h>
#include <ert/util/subst_list.h>

int main(int argc, char ** argv) {
  subst_func_pool_type * subst_func_pool = subst_func_pool_alloc();
  subst_list_type * subst_list = subst_list_alloc(subst_func_pool);
  ert_workflow_list_type * workflow_list = ert_workflow_list_alloc(subst_list);
  qc_module_type * qc_module = qc_module_alloc(workflow_list, "");
  qc_module_set_runpath_list_file(qc_module, "testing");
  test_assert_string_equal("testing", qc_module_get_runpath_list_file(qc_module));

  qc_module_set_runpath_list_basepath(qc_module, "Ensemble");

  char * path = util_alloc_sprintf("Ensemble/%s", ".ert_runpath_list");
  test_assert_string_equal(path, qc_module_get_runpath_list_file(qc_module));
  free(path);
  free(subst_func_pool);
  free(subst_list);
  free(workflow_list);
  free(qc_module);

  exit(0);
}

