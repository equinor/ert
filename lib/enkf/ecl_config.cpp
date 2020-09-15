/*
 Copyright (C) 2011  Equinor ASA, Norway.

 The file 'ecl_config.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <time.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include <unordered_map>

#include <ert/util/util.h>
#include <ert/util/parser.h>
#include <ert/res_util/ui_return.hpp>
#include <ert/res_util/path_fmt.hpp>


#include <ert/config/config_parser.hpp>
#include <ert/config/config_content.hpp>
#include <ert/config/config_schema_item.hpp>

#include <ert/ecl/ecl_grid.h>
#include <ert/ecl/ecl_sum.h>
#include <ert/ecl/ecl_io_config.h>
#include <ert/ecl/ecl_util.h>

#include <ert/enkf/enkf_util.hpp>
#include <ert/enkf/ecl_config.hpp>
#include <ert/enkf/config_keys.hpp>
#include <ert/enkf/ecl_refcase_list.hpp>
#include <ert/enkf/enkf_defaults.hpp>
#include <ert/enkf/model_config.hpp>

/**
 This file implements a struct which holds configuration information
 needed to run ECLIPSE.

 Pointers to the fields in this structure are passed on to e.g. the
 enkf_state->shared_info object, but this struct is the *OWNER* of
 this information, and hence responsible for booting and deleting
 these objects.

 Observe that the distinction of what goes in model_config, and what
 goes in ecl_config is not entirely clear.
 */

struct ecl_config_struct
{
  ecl_io_config_type * io_config;       /* This struct contains information of whether the eclipse files should be formatted|unified|endian_fliped */
  char * data_file;                     /* Eclipse data file. */
  time_t start_date;                    /* The start date of the ECLIPSE simulation - parsed from the data_file. */
  time_t end_date;                      /* An optional date value which can be used to check if the ECLIPSE simulation has been 'long enough'. */
  ecl_refcase_list_type * refcase_list;
  ecl_grid_type * grid;                 /* The grid which is active for this model. */
  char * schedule_prediction_file;      /* Name of schedule prediction file - observe that this is internally handled as a gen_kw node. */
  int last_history_restart;
  bool can_restart;                     /* Have we found the <INIT> tag in the data file? */
  bool have_eclbase;
  int num_cpu;                          /* We should parse the ECLIPSE data file and determine how many cpus this eclipse file needs. */
  ert_ecl_unit_enum unit_system;        /* Either metric, field or lab */
};

/*****************************************************************/

/**
 With this function we try to determine whether ECLIPSE is active
 for this case, i.e. if ECLIPSE is part of the forward model. This
 should ideally be inferred from the FORWARD model, but what we do
 here is just to check if the core field ->eclbase or ->data_file
 have been set. If they are both equal to NULL we assume that
 ECLIPSE is not active and return false, otherwise we return true.
 */

bool ecl_config_active(const ecl_config_type * config)
{
  if (config->have_eclbase)
    return true;

  if (config->data_file)
    return true;

  return false;
}

  /**
   Could look up the sched_file instance directly - because the
   ecl_config will never be the owner of a file with predictions.
   */

int ecl_config_get_last_history_restart(const ecl_config_type * ecl_config)
{
  return ecl_config->last_history_restart;
}

bool ecl_config_can_restart(const ecl_config_type * ecl_config)
{
  return ecl_config->can_restart;
}

void ecl_config_assert_restart(const ecl_config_type * ecl_config)
{
  if (!ecl_config_can_restart(ecl_config))
  {
    fprintf(stderr, "** Warning - tried to restart case which is not properly set up for restart.\n");
    fprintf(stderr, "** Need <INIT> in datafile and INIT_SECTION keyword in config file.\n");
    util_exit("%s: exiting \n", __func__);
  }
}

ui_return_type * ecl_config_validate_data_file(const ecl_config_type * ecl_config, const char * data_file) {    
  if (util_file_exists(data_file))
    return ui_return_alloc(UI_RETURN_OK);
  else {
    ui_return_type * ui_return = ui_return_alloc(UI_RETURN_FAIL);
    char * error_msg = util_alloc_sprintf("File not found:%s" , data_file);
    ui_return_add_error(ui_return , error_msg);
    free( error_msg );
    return ui_return;
  }
}


void ecl_config_set_data_file(ecl_config_type * ecl_config, const char * data_file) {
  ecl_config->data_file = util_realloc_string_copy(ecl_config->data_file, data_file);
  {
    FILE * stream = util_fopen(ecl_config->data_file, "r");
    basic_parser_type * parser = basic_parser_alloc(NULL, NULL, NULL, NULL, "--", "\n");
    char * init_tag = enkf_util_alloc_tagged_string("INIT");

    ecl_config->can_restart = basic_parser_fseek_string(parser, stream, init_tag, false, true);

    free(init_tag);
    basic_parser_free(parser);
    fclose(stream);
  }
  ecl_config->start_date = ecl_util_get_start_date(ecl_config->data_file);
  ecl_config->num_cpu = ecl_util_get_num_cpu(ecl_config->data_file);
  ecl_config->unit_system = ecl_util_get_unit_set(ecl_config->data_file);
}


const char * ecl_config_get_data_file(const ecl_config_type * ecl_config)
{
  return ecl_config->data_file;
}

time_t ecl_config_get_start_date(const ecl_config_type * ecl_config)
{
  return ecl_config->start_date;
}

time_t ecl_config_get_end_date(const ecl_config_type * ecl_config)
{
  return ecl_config->end_date;
}

static void ecl_config_set_end_date(ecl_config_type * ecl_config, time_t end_date)
{
  ecl_config->end_date = end_date;
}

int ecl_config_get_num_cpu(const ecl_config_type * ecl_config)
{
  return ecl_config->num_cpu;
}

const char * ecl_config_get_schedule_prediction_file(const ecl_config_type * ecl_config)
{
  return ecl_config->schedule_prediction_file;
}

/**
   Observe: The real schedule prediction functionality is implemented
   as a special GEN_KW node in ensemble_config.
 */

void ecl_config_set_schedule_prediction_file(ecl_config_type * ecl_config, const char * schedule_prediction_file)
{
  ecl_config->schedule_prediction_file = util_realloc_string_copy(ecl_config->schedule_prediction_file, schedule_prediction_file);
}

ui_return_type * ecl_config_validate_eclbase(const ecl_config_type * ecl_config, const char * eclbase_fmt) {
  if (ecl_util_valid_basename_fmt(eclbase_fmt))
    return ui_return_alloc(UI_RETURN_OK);
  else {
    ui_return_type * ui_return = ui_return_alloc(UI_RETURN_FAIL);
    {
      char * error_msg = util_alloc_sprintf("The format string: %s was invalid as ECLBASE format", eclbase_fmt);
      ui_return_add_error(ui_return, error_msg);
      free(error_msg);
    }
    ui_return_add_help(ui_return , "The eclbase format must have all characters in the same case,");
    ui_return_add_help(ui_return , "in addition it can contain a %d specifier which will be");
    ui_return_add_help(ui_return , "with the realization number.");

    return ui_return;
  }
}


/**
 Can be called with @refcase == NULL - which amounts to clearing the
 current refcase.
*/
bool ecl_config_load_refcase(ecl_config_type * ecl_config, const char * refcase)
{
  return ecl_refcase_list_set_default(ecl_config->refcase_list, refcase);
}


ui_return_type * ecl_config_validate_refcase( const ecl_config_type * ecl_config , const char * refcase) {
  if (ecl_sum_case_exists( refcase ))
    return ui_return_alloc( UI_RETURN_OK );
  else {
    ui_return_type * ui_return = ui_return_alloc( UI_RETURN_FAIL );
    char * error_msg = util_alloc_sprintf( "Could not load summary case from:%s \n",refcase);
    ui_return_add_error( ui_return , error_msg );
    free( error_msg );
    return ui_return;
  }
}


/**
 Will return NULL if no refcase is set.
 */
const char * ecl_config_get_refcase_name(const ecl_config_type * ecl_config)
{
  const ecl_sum_type * refcase = ecl_refcase_list_get_default(ecl_config->refcase_list);
  if (refcase == NULL )
    return NULL ;
  else
    return ecl_sum_get_case(refcase);

}

static ecl_config_type * ecl_config_alloc_empty(void)
{
  ecl_config_type * ecl_config = new ecl_config_type();

  ecl_config->io_config = ecl_io_config_alloc(DEFAULT_FORMATTED, DEFAULT_UNIFIED, DEFAULT_UNIFIED);
  ecl_config->have_eclbase = false;
  ecl_config->num_cpu = 1; /* This must get a valid default in case no ECLIPSE datafile is provided. */
  ecl_config->unit_system = ECL_METRIC_UNITS;
  ecl_config->data_file = NULL;
  ecl_config->grid = NULL;
  ecl_config->can_restart = false;
  ecl_config->start_date = -1;
  ecl_config->end_date = -1;
  ecl_config->schedule_prediction_file = NULL;
  ecl_config->refcase_list = ecl_refcase_list_alloc();

  return ecl_config;
}

ecl_config_type * ecl_config_alloc(const config_content_type * config_content) {
  ecl_config_type * ecl_config = ecl_config_alloc_empty();

  if(config_content)
    ecl_config_init(ecl_config, config_content);

  return ecl_config;
}

ecl_config_type * ecl_config_alloc_full(bool have_eclbase, 
                                        char * data_file, 
                                        ecl_grid_type * grid,
                                        char * refcase_default,
                                        stringlist_type * ref_case_list,
                                        time_t end_date,
                                        char * sched_prediction_file
                                        ) {
  ecl_config_type * ecl_config = ecl_config_alloc_empty();
  ecl_config->have_eclbase = have_eclbase;
  ecl_config->grid = grid;
  if (data_file != NULL) {
    ecl_config_set_data_file(ecl_config, data_file);
  }

  for (int i = 0; i < stringlist_get_size(ref_case_list); i++)
  {
    ecl_refcase_list_add_matching(ecl_config->refcase_list, stringlist_safe_iget(ref_case_list, i));
  }
  if (refcase_default)
    ecl_refcase_list_set_default(ecl_config->refcase_list, refcase_default);

  ecl_config->end_date = end_date;
  if (sched_prediction_file)
    ecl_config->schedule_prediction_file = util_alloc_string_copy(sched_prediction_file);
  
  return ecl_config;
}

static void handle_has_eclbase_key(ecl_config_type * ecl_config,
                                   const config_content_type * config) {
  /*
     The eclbase is not internalized here; here we only flag that the
     ECLBASE keyword has been present in the configuration. The
     actualt value is internalized as a job_name in the model_config.
   */

  if (config_content_has_item(config, ECLBASE_KEY)) {
    ui_return_type * ui_return = ecl_config_validate_eclbase(ecl_config, config_content_iget(config, ECLBASE_KEY, 0, 0));
    if (ui_return_get_status(ui_return) == UI_RETURN_OK)
      ecl_config->have_eclbase = true;
    else
      util_abort("%s: failed to set eclbase format. Error:%s\n", __func__ , ui_return_get_last_error(ui_return));
    ui_return_free(ui_return);
  }
}

static void handle_has_data_file_key(ecl_config_type * ecl_config,
                                     const config_content_type * config) {
  const char * data_file = config_content_get_value_as_abspath(config,
                                                               DATA_FILE_KEY);
  ui_return_type * ui_return = ecl_config_validate_data_file(ecl_config,
                                                             data_file);
  if (ui_return_get_status( ui_return ) == UI_RETURN_OK)
    ecl_config_set_data_file(ecl_config, data_file);
  else
    util_abort("%s: problem setting ECLIPSE data file (%s)\n",
               __func__, ui_return_get_last_error(ui_return));
  ui_return_free(ui_return);
}

static void handle_has_grid_key(ecl_config_type * ecl_config,
                                const config_content_type * config) {
  const char * grid_file = config_content_get_value_as_abspath(config, GRID_KEY);

  ui_return_type * ui_return = ecl_config_validate_grid(ecl_config, grid_file);
  if (ui_return_get_status(ui_return) == UI_RETURN_OK)
    ecl_config_set_grid(ecl_config, grid_file );
  else
    util_abort("%s: failed to set grid file:%s  Error:%s \n",
               __func__,
               grid_file,
               ui_return_get_last_error(ui_return));

  ui_return_free(ui_return);
}


static void handle_has_refcase_key(ecl_config_type * ecl_config,
                                   const config_content_type * config) {
  const char * refcase_path = config_content_get_value_as_abspath(config,
                                                                  REFCASE_KEY);

  if (!ecl_config_load_refcase(ecl_config, refcase_path))
    fprintf(stderr, "** Warning: loading refcase:%s failed \n", refcase_path);
}

static void handle_has_refcase_list_key(ecl_config_type * ecl_config,
                                        const config_content_type * config) {
  config_content_item_type * item = config_content_get_item(config, REFCASE_LIST_KEY);
  for (int i = 0; i < config_content_item_get_size(item); i++) {
    config_content_node_type * node = config_content_item_iget_node(item, i);
    for (int j = 0; j < config_content_node_get_size(node); j++) {
      const char * refcase_list_path = config_content_node_iget_as_abspath(node, j);
      ecl_refcase_list_add_matching(ecl_config->refcase_list, refcase_list_path);
    }
  }
}

static void handle_has_end_date_key(ecl_config_type * ecl_config,
                                    const config_content_type * config) {
  const char * date_string = config_content_get_value(config, END_DATE_KEY);
  time_t end_date;
  if (util_sscanf_date_utc(date_string, &end_date))
    ecl_config_set_end_date(ecl_config, end_date);
  else
    fprintf(stderr, "** WARNING **: Failed to parse %s as a date - should be in format dd/mm/yyyy \n", date_string);
}

static void handle_has_schedule_prediction_file_key(ecl_config_type * ecl_config,
                                                    const config_content_type * config) {
  const config_content_item_type * pred_item = config_content_get_item(
    config,
    SCHEDULE_PREDICTION_FILE_KEY
    );

  config_content_node_type * pred_node = config_content_item_get_last_node(pred_item);
  const char * template_file = config_content_node_iget_as_path(pred_node, 0);
  ecl_config_set_schedule_prediction_file(ecl_config, template_file);
}





void ecl_config_init(ecl_config_type * ecl_config, const config_content_type * config)
{
  if (config_content_has_item(config, ECLBASE_KEY))
    handle_has_eclbase_key(ecl_config, config);

  if (config_content_has_item(config, DATA_FILE_KEY))
    handle_has_data_file_key(ecl_config, config);

  if (config_content_has_item(config, GRID_KEY))
    handle_has_grid_key(ecl_config, config);

  if (config_content_has_item(config, REFCASE_KEY))
    handle_has_refcase_key(ecl_config, config);

  if (config_content_has_item(config, REFCASE_LIST_KEY))
    handle_has_refcase_list_key(ecl_config, config);


  if (ecl_config->can_restart)
    fprintf(stderr,
            "** Warning: The ECLIPSE data file contains a <INIT> section, the support\n"
            "            for this functionality has been removed. libres will not\n"
            "            be able to properly initialize the ECLIPSE MODEL.\n"
            );
    /**
     This is a hard error - the datafile contains <INIT>, however
     the config file does NOT contain INIT_SECTION, i.e. we have
     no information to fill in for the <INIT> section. This case
     will not be able to initialize an ECLIPSE model, and that is
     broken behaviour.
     */

  /*
   The user has not supplied a INIT_SECTION keyword whatsoever,
   this essentially means that we can not restart - because:

   1. The EQUIL section must be inlined in the DATAFILE without any
   special markup.

   2. ECLIPSE will fail hard if the datafile contains both an EQUIL
   section and a restart statement, and when we have not marked
   the EQUIL section specially with the INIT_SECTION keyword it
   is impossible for ERT to dynamically change between a
   datafile with initialisation and a datafile for restart.

   IFF the user has no intentitions of any form of restart, this is
   perfectly legitemate.
   */
  if (config_content_has_item(config, END_DATE_KEY))
    handle_has_end_date_key(ecl_config, config);

  if (config_content_has_item(config, SCHEDULE_PREDICTION_FILE_KEY))
    handle_has_schedule_prediction_file_key(ecl_config, config);
}



void ecl_config_free(ecl_config_type * ecl_config)
{
  ecl_io_config_free(ecl_config->io_config);
  free(ecl_config->data_file);
  free(ecl_config->schedule_prediction_file);

  if (ecl_config->grid != NULL )
    ecl_grid_free(ecl_config->grid);

  ecl_refcase_list_free(ecl_config->refcase_list);

  delete ecl_config;
}


ecl_grid_type * ecl_config_get_grid(const ecl_config_type * ecl_config)
{
  return ecl_config->grid;
}

const char * ecl_config_get_gridfile(const ecl_config_type * ecl_config)
{
  if (ecl_config->grid == NULL )
    return NULL ;
  else
    return ecl_grid_get_name(ecl_config->grid);
}

/**
   The ecl_config object isolated supports run-time changing of the
   grid, however this does not (in general) apply to the system as a
   whole. Other objects which internalize pointers (i.e. field_config
   objects) to an ecl_grid_type instance will be left with dangling
   pointers; and things will probably die an ugly death. So - changing
   grid runtime should be done with extreme care.
*/

void ecl_config_set_grid(ecl_config_type * ecl_config, const char * grid_file)
{
  if (ecl_config->grid != NULL )
    ecl_grid_free(ecl_config->grid);
  ecl_config->grid = ecl_grid_alloc(grid_file);
}

ui_return_type * ecl_config_validate_grid( const ecl_config_type * ecl_config , const char * grid_file ) {
  ui_return_type * ui_return;
  if (util_file_exists( grid_file )) {
    ecl_file_enum file_type = ecl_util_get_file_type( grid_file , NULL , NULL );
    if ((file_type == ECL_EGRID_FILE) || (file_type == ECL_GRID_FILE))
      ui_return =  ui_return_alloc( UI_RETURN_OK );
    else {
      ui_return =  ui_return_alloc( UI_RETURN_FAIL );
      ui_return_add_error( ui_return , "Input argument is not a GRID/EGRID file");
    }
  } else {
    ui_return =  ui_return_alloc( UI_RETURN_FAIL );
    ui_return_add_error( ui_return , "Input argument does not exist.");
  }
  return ui_return;
}



ecl_refcase_list_type * ecl_config_get_refcase_list(const ecl_config_type * ecl_config)
{
  return ecl_config->refcase_list;
}

const ecl_sum_type * ecl_config_get_refcase(const ecl_config_type * ecl_config)
{
  return ecl_refcase_list_get_default(ecl_config->refcase_list);
}

bool ecl_config_has_refcase(const ecl_config_type * ecl_config)
{
  const ecl_sum_type * refcase = ecl_config_get_refcase(ecl_config);
  if (refcase)
    return true;
  else
    return false;
}

bool ecl_config_get_formatted(const ecl_config_type * ecl_config)
{
  return ecl_io_config_get_formatted(ecl_config->io_config);
}
bool ecl_config_get_unified_restart(const ecl_config_type * ecl_config)
{
  return ecl_io_config_get_unified_restart(ecl_config->io_config);
}

bool ecl_config_have_eclbase(const ecl_config_type * ecl_config) {
  return ecl_config->have_eclbase;
}


void ecl_config_add_config_items(config_parser_type * config)
{
  config_schema_item_type * item;

  /*
   Observe that SCHEDULE_PREDICTION_FILE - which is implemented as a
   GEN_KW is added in ensemble_config.c
  */

  item = config_add_schema_item(config, ECLBASE_KEY, false);
  config_schema_item_set_argc_minmax(item, 1, 1);

  item = config_add_schema_item(config, DATA_FILE_KEY, false);
  config_schema_item_set_argc_minmax(item, 1, 1);
  config_schema_item_iset_type(item, 0, CONFIG_EXISTING_PATH);

  item = config_add_schema_item(config, REFCASE_KEY, false);
  config_schema_item_set_argc_minmax(item, 1, 1);
  config_schema_item_iset_type(item, 0, CONFIG_PATH);

  item = config_add_schema_item(config, REFCASE_LIST_KEY, false);
  config_schema_item_set_default_type(item, CONFIG_PATH);

  item = config_add_schema_item(config, GRID_KEY, false);
  config_schema_item_set_argc_minmax(item, 1, 1);
  config_schema_item_iset_type(item, 0, CONFIG_EXISTING_PATH);

  item = config_add_schema_item(config, END_DATE_KEY, false);
  config_schema_item_set_argc_minmax(item, 1, 1);
}


/* Units as specified in the ECLIPSE technical manual */
const char * ecl_config_get_depth_unit(const ecl_config_type * ecl_config)
{
  switch(ecl_config->unit_system) {
  case ECL_METRIC_UNITS:
    return "M";
  case ECL_FIELD_UNITS:
    return "FT";
  case ECL_LAB_UNITS:
    return "CM";
  default:
    util_abort("%s: unit system enum value:%d not recognized \n",__func__ , ecl_config->unit_system);
    return NULL;
  }
}


const char * ecl_config_get_pressure_unit(const ecl_config_type * ecl_config)
{
  switch(ecl_config->unit_system) {
  case ECL_METRIC_UNITS:
    return "BARSA";
  case ECL_FIELD_UNITS:
    return "PSIA";
  case ECL_LAB_UNITS:
    return "ATMA";
  default:
    util_abort("%s: unit system enum value:%d not recognized \n",__func__ , ecl_config->unit_system);
    return NULL;
  }
}
