/*
   Copyright (C) 2014  Equinor ASA, Norway.

   The file 'enkf_export_inactive_cells.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <stdlib.h>
#include <stdio.h>

#include <ert/util/test_util.h>
#include <ert/enkf/ert_test_context.hpp>

#include <ert/rms/rms_util.hpp>

#include <ert/enkf/field.hpp>


void check_exported_data(const char * exported_file,
                                const char * init_file,
                                field_file_format_type file_type,
                                const field_config_type * field_config,
                                const field_type * field,
                                int nx,
                                int ny,
                                int nz) {

  FILE * original_stream                    = NULL;
  ecl_kw_type * kw_original                 = NULL;
  FILE * exported_stream                    = NULL;
  ecl_kw_type * kw_exported                 = NULL;
  field_type * exported_field               = NULL;
  field_config_type * exported_field_config = NULL;

  {
    if (init_file) {
      original_stream = util_fopen( init_file , "r");
      kw_original = ecl_kw_fscanf_alloc_grdecl_dynamic( original_stream , field_config_get_key(field_config) , ECL_DOUBLE );
    }

    if (ECL_GRDECL_FILE == file_type) {
      exported_stream = util_fopen( exported_file , "r");
      kw_exported     = ecl_kw_fscanf_alloc_grdecl_dynamic( exported_stream , field_config_get_key(field_config) , ECL_DOUBLE );
    } else if (RMS_ROFF_FILE == file_type) {
      ecl_grid_type * grid  = field_config_get_grid(field_config);
      exported_field_config = field_config_alloc_empty(field_config_get_key(field_config), grid, NULL, true);
      exported_field        = field_alloc(exported_field_config);

      bool keep_inactive = true;
      field_fload_rms(exported_field, exported_file, keep_inactive);
    }
  }


  {
    int k, j, i = 0;

    for (k=0; k < nz; k++) {
      for (j=0; j < ny; j++) {
        for (i=0; i < nx; i++) {
          bool active           = field_config_active_cell(field_config, i, j, k);
          double field_value    = active ? field_ijk_get_double(field, i, j, k) : 0.0;
          int global_index      = field_config_global_index(field_config , i , j , k);
          double exported_value = 0.0;
          if (ECL_GRDECL_FILE == file_type)
            exported_value = ecl_kw_iget_as_double(kw_exported, global_index);
          else if (RMS_ROFF_FILE == file_type) {
            exported_value = field_ijk_get_double(exported_field, i, j, k);
          }
          double initial_value  = init_file ? ecl_kw_iget_as_double(kw_original, global_index) : 0.0;

          if (active)
            test_assert_double_equal(field_value, exported_value);
          else if (init_file)
            test_assert_double_equal(initial_value, exported_value);
          else if (file_type == RMS_ROFF_FILE)
            test_assert_double_equal(RMS_INACTIVE_DOUBLE, exported_value);
          else
            test_assert_double_equal(0.0, exported_value);
        }
      }
    }
  }


  if (init_file) {
    fclose(original_stream);
    ecl_kw_free(kw_original);
  }

  if (ECL_GRDECL_FILE == file_type) {
    fclose(exported_stream);
    ecl_kw_free(kw_exported);
  } else
    field_free(exported_field);
}



void forward_initialize_node(enkf_main_type * enkf_main, const char * init_file, enkf_node_type * field_node) {
  enkf_fs_type * fs           =  enkf_main_get_fs(enkf_main);
  subst_list_type * subst_list = subst_list_alloc(NULL);
  {
    const int ens_size          = enkf_main_get_ensemble_size( enkf_main );
    bool_vector_type * iactive  = bool_vector_alloc(ens_size, true);
    const path_fmt_type * runpath_fmt = model_config_get_runpath_fmt( enkf_main_get_model_config( enkf_main ));
    ert_run_context_type * run_context = ert_run_context_alloc_INIT_ONLY( fs, INIT_CONDITIONAL , iactive, runpath_fmt, subst_list , 0 );

    enkf_main_create_run_path(enkf_main , run_context );
    bool_vector_free(iactive);
    ert_run_context_free( run_context );
  }

  {
    int iens                = 0;
    enkf_state_type * state = enkf_main_iget_state( enkf_main , iens );

    run_arg_type  * run_arg = run_arg_alloc_ENSEMBLE_EXPERIMENT( "RUN_ID", fs , 0 ,0 , "simulations/run0", "path", subst_list);

    ensemble_config_forward_init( enkf_state_get_ensemble_config( state ) , run_arg);
  }

  subst_list_free(subst_list);
}



int main(int argc , char ** argv) {
  enkf_main_install_SIGNALS();

  const char * config_file = argv[1];
  const char * init_file   = argv[2];
  const char * key         = "PORO";
  int iens                 = 0;

  ert_test_context_type      * test_context    = ert_test_context_alloc("ExportInactiveCellsTest" , config_file);
  enkf_main_type             * enkf_main       = ert_test_context_get_main(test_context);
  enkf_fs_type               * fs              = enkf_main_get_fs(enkf_main);
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  enkf_config_node_type      * config_node     = ensemble_config_get_node(ensemble_config , key);
  const field_config_type * field_config = (const field_config_type *)enkf_config_node_get_ref( config_node );
  enkf_node_type             * field_node      = enkf_node_alloc( config_node );
  field_type                 * field           = (field_type *) enkf_node_value_ptr(field_node);

  {
    forward_initialize_node(enkf_main, init_file, field_node);
    node_id_type node_id = {.report_step = 0 , .iens = iens };
    test_assert_true(enkf_node_try_load(field_node , fs , node_id));
    field_scale(field, 3.0);
  }

  int nx,ny,nz;
  field_config_get_dims(field_config , &nx , &ny , &nz);
  const char * export_file_grdecl = "my_test_dir/exported_field_test_file_grdecl";
  const char * export_file_roff   = "my_test_dir/exported_field_test_file_roff";
  field_file_format_type file_type;
  model_config_type * mc = enkf_main_get_model_config(enkf_main);
  path_fmt_type * runpath_fmt = model_config_get_runpath_fmt(mc);
  const char * found_init_file = enkf_config_node_get_FIELD_fill_file(config_node, runpath_fmt);
  {
    file_type = ECL_GRDECL_FILE;
    field_export(field, export_file_grdecl, NULL, file_type, false, found_init_file);
    check_exported_data(export_file_grdecl, init_file, file_type, field_config, field, nx, ny, nz);
  }
  {
    file_type = RMS_ROFF_FILE;
    field_export(field, export_file_roff, NULL, file_type, false, found_init_file);
    check_exported_data(export_file_roff, init_file, file_type, field_config, field, nx, ny, nz);
  }

  found_init_file = NULL;
  {
    file_type = ECL_GRDECL_FILE;
    field_export(field, export_file_grdecl, NULL, file_type, false, found_init_file);
    check_exported_data(export_file_grdecl, found_init_file, file_type, field_config, field, nx, ny, nz);
  }
  {
    file_type = RMS_ROFF_FILE;
    field_export(field, export_file_roff, NULL, file_type, false, found_init_file);
    check_exported_data(export_file_roff, found_init_file, file_type, field_config, field, nx, ny, nz);
  }



  ert_test_context_free(test_context);
  exit(0);
}

