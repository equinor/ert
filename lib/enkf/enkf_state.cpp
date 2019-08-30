/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'enkf_state.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <sys/types.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <pthread.h>

#include <ert/util/hash.h>
#include <ert/util/util.h>
#include <ert/res_util/arg_pack.hpp>
#include <ert/util/stringlist.h>
#include <ert/util/node_ctype.h>
#include <ert/util/timer.h>
#include <ert/util/time_t_vector.h>
#include <ert/util/rng.h>
#include <ert/res_util/subst_list.hpp>

#include <ert/ecl/fortio.h>
#include <ert/ecl/ecl_kw.h>
#include <ert/ecl/ecl_io_config.h>
#include <ert/ecl/ecl_file.h>
#include <ert/ecl/ecl_util.h>
#include <ert/ecl/ecl_sum.h>
#include <ert/ecl/ecl_endian_flip.h>

#include <ert/sched/sched_file.hpp>

#include <ert/job_queue/environment_varlist.hpp>
#include <ert/job_queue/forward_model.hpp>
#include <ert/job_queue/job_queue.hpp>
#include <ert/job_queue/queue_driver.hpp>
#include <ert/job_queue/ext_joblist.hpp>

#include <ert/enkf/enkf_node.hpp>
#include <ert/enkf/enkf_state.hpp>
#include <ert/enkf/enkf_types.hpp>
#include <ert/enkf/field.hpp>
#include <ert/enkf/field_config.hpp>
#include <ert/enkf/gen_kw.hpp>
#include <ert/enkf/summary.hpp>
#include <ert/enkf/gen_data.hpp>
#include <ert/enkf/enkf_fs.hpp>
#include <ert/enkf/ensemble_config.hpp>
#include <ert/enkf/model_config.hpp>
#include <ert/enkf/site_config.hpp>
#include <ert/enkf/ecl_config.hpp>
#include <ert/enkf/ert_template.hpp>
#include <ert/enkf/enkf_defaults.hpp>
#include <ert/enkf/state_map.hpp>
#include <ert/res_util/res_log.hpp>
#include <ert/enkf/run_arg.hpp>
#include <ert/enkf/summary_key_matcher.hpp>
#include <ert/enkf/forward_load_context.hpp>
#include <ert/enkf/enkf_config_node.hpp>
#include <ert/enkf/callback_arg.hpp>

#define  ENKF_STATE_TYPE_ID 78132





/**
   This struct contains various objects which the enkf_state needs
   during operation, which the enkf_state_object *DOES NOT* own. The
   struct only contains pointers to objects owned by (typically) the
   enkf_main object.

   If the enkf_state object writes to any of the objects in this
   struct that can be considered a serious *BUG*.

   The elements in this struct should not change during the
   application lifetime?
*/

typedef struct shared_info_struct {
  model_config_type           * model_config;      /* .... */
  ext_joblist_type            * joblist;           /* The list of external jobs which are installed - and *how* they should be run (with Python code) */
  const site_config_type      * site_config;
  ert_templates_type          * templates;
  const ecl_config_type       * ecl_config;
} shared_info_type;






/*****************************************************************/

struct enkf_state_struct {
  UTIL_TYPE_ID_DECLARATION;
  hash_type             * node_hash;
  ensemble_config_type  * ensemble_config;         /* The config nodes for the enkf_node objects contained in node_hash. */
  shared_info_type      * shared_info;             /* Pointers to shared objects which is needed by the enkf_state object (read only). */
  int                     __iens;
};

/*****************************************************************/


static UTIL_SAFE_CAST_FUNCTION( enkf_state , ENKF_STATE_TYPE_ID )


static shared_info_type * shared_info_alloc(const site_config_type * site_config , model_config_type * model_config, const ecl_config_type * ecl_config , ert_templates_type * templates) {
  shared_info_type * shared_info = (shared_info_type *)util_malloc(sizeof * shared_info );
  shared_info->joblist      = site_config_get_installed_jobs( site_config );
  shared_info->site_config  = site_config;
  shared_info->model_config = model_config;
  shared_info->templates    = templates;
  shared_info->ecl_config   = ecl_config;
  return shared_info;
}


static void shared_info_free(shared_info_type * shared_info) {
  /**
      Adding something here is a BUG - this object does
      not own anything.
  */
  free( shared_info );
}





/*****************************************************************/
/** Helper classes complete - starting on the enkf_state proper object. */
/*****************************************************************/


/*
  This function does not acces the nodes of the enkf_state object.
*/
void enkf_state_initialize(enkf_state_type * enkf_state , rng_type * rng, enkf_fs_type * fs , const stringlist_type * param_list, init_mode_type init_mode) {
  if (init_mode != INIT_NONE) {
    int iens = enkf_state_get_iens( enkf_state );
    state_map_type * state_map = enkf_fs_get_state_map( fs );
    realisation_state_enum current_state = state_map_iget(state_map, iens);
    if ((current_state == STATE_PARENT_FAILURE) && (init_mode != INIT_FORCE))
      return;
    else {
      const ensemble_config_type * ensemble_config = enkf_state->ensemble_config;
      for (int ip = 0; ip < stringlist_get_size(param_list); ip++) {
        const enkf_config_node_type * config_node = ensemble_config_get_node( ensemble_config , stringlist_iget(param_list, ip));
        enkf_node_type * param_node = enkf_node_alloc( config_node );
        node_id_type node_id = { .report_step = 0, .iens = iens };
        bool has_data = enkf_node_has_data(param_node, fs, node_id);

        if ((init_mode == INIT_FORCE) || (has_data == false) || (current_state == STATE_LOAD_FAILURE)) {
          if (enkf_node_initialize(param_node, iens, rng))
            enkf_node_store(param_node, fs, true, node_id);
        }

        enkf_node_free( param_node );
      }
      state_map_update_matching(state_map , iens , STATE_UNDEFINED | STATE_LOAD_FAILURE , STATE_INITIALIZED);
      enkf_fs_fsync(fs);
    }
  }
}







int enkf_state_get_iens(const enkf_state_type * enkf_state) {
  return enkf_state->__iens;
}





static void enkf_state_add_subst_kw( enkf_state_type * enkf_state, const char * key, const char * value, const char * doc) {

}




static void enkf_state_add_nodes( enkf_state_type * enkf_state, const ensemble_config_type * ensemble_config) {
  stringlist_type * container_keys = stringlist_alloc_new();
  stringlist_type * keylist  = ensemble_config_alloc_keylist(ensemble_config);
  int keys        = stringlist_get_size(keylist);

  // 1: Add all regular nodes
  for (int ik = 0; ik < keys; ik++) {
    const char * key = stringlist_iget(keylist, ik);
    const enkf_config_node_type * config_node = ensemble_config_get_node(ensemble_config , key);
    if (enkf_config_node_get_impl_type( config_node ) == CONTAINER) {
      stringlist_append_copy( container_keys , key );
    } else
      enkf_state_add_node(enkf_state , key , config_node);
  }

  // 2: Add container nodes - must ensure that all other nodes have
  //    been added already (this implies that containers of containers
  //    will be victim of hash retrieval order problems ....

  for (int ik = 0; ik < stringlist_get_size( container_keys ); ik++) {
    const char * key = stringlist_iget(container_keys, ik);
    const enkf_config_node_type * config_node = ensemble_config_get_node(ensemble_config , key);
    enkf_state_add_node( enkf_state , key , config_node );
  }

  stringlist_free(keylist);
  stringlist_free( container_keys );
}


enkf_state_type * enkf_state_alloc(int iens,
                                   rng_type                  * rng ,
                                   model_config_type         * model_config,
                                   ensemble_config_type      * ensemble_config,
                                   const site_config_type    * site_config,
                                   const ecl_config_type     * ecl_config,
                                   ert_templates_type        * templates) {

  enkf_state_type * enkf_state = (enkf_state_type *)util_malloc(sizeof *enkf_state );
  UTIL_TYPE_ID_INIT( enkf_state , ENKF_STATE_TYPE_ID );

  enkf_state->ensemble_config   = ensemble_config;
  enkf_state->shared_info       = shared_info_alloc(site_config , model_config , ecl_config , templates);
  enkf_state->node_hash         = hash_alloc();

  /**
     Adding all the subst_kw keywords here, with description. Listing
     all of them here in one go guarantees that we have control over
     the ordering (which is interesting because the substititions are
     done in a cascade like fashion). The user defined keywords are
     added first, so that these can refer to the built in keywords.
  */

  enkf_state_add_subst_kw(enkf_state , "RUNPATH"       , "---" , "The absolute path of the current forward model instance. ");
  enkf_state_add_subst_kw(enkf_state , "IENS"          , "---" , "The realisation number for this realization.");
  enkf_state_add_subst_kw(enkf_state , "IENS4"         , "---" , "The realization number for this realization - formated with %04d.");
  enkf_state_add_subst_kw(enkf_state , "ECLBASE"       , "---" , "The ECLIPSE basename for this realization.");
  enkf_state_add_subst_kw(enkf_state , "ECL_BASE"      , "---" , "Depreceated - use ECLBASE instead.");
  enkf_state_add_subst_kw(enkf_state , "SMSPEC"        , "---" , "The ECLIPSE SMSPEC file for this realization.");
  enkf_state_add_subst_kw(enkf_state , "TSTEP1"        , "---" , "The initial report step for this simulation.");
  enkf_state_add_subst_kw(enkf_state , "TSTEP2"        , "---" , "The final report step for this simulation.");
  enkf_state_add_subst_kw(enkf_state , "TSTEP1_04"     , "---" , "The initial report step for this simulation - formated with %04d.");
  enkf_state_add_subst_kw(enkf_state , "TSTEP2_04"     , "---" , "The final report step for this simulation - formated withh %04d.");
  enkf_state_add_subst_kw(enkf_state , "RESTART_FILE1" , "---" , "The ECLIPSE restart file this simulation starts with.");
  enkf_state_add_subst_kw(enkf_state , "RESTART_FILE2" , "---" , "The ECLIPSE restart file this simulation should end with.");

  enkf_state->__iens = iens;
  enkf_state_add_nodes( enkf_state , ensemble_config );

  return enkf_state;
}









static void enkf_state_log_GEN_DATA_load( const enkf_node_type * enkf_node , int report_step , forward_load_context_type * load_context) {
  if (forward_load_context_accept_messages(load_context)) {
    char * load_file = enkf_config_node_alloc_infile(enkf_node_get_config( enkf_node ) , report_step);
    int data_size = gen_data_get_size( (const gen_data_type * ) enkf_node_value_ptr( enkf_node ));
    char * msg = util_alloc_sprintf("Loaded GEN_DATA:%s instance for step:%d from file:%s size:%d" ,
                                    enkf_node_get_key( enkf_node ) ,
                                    report_step ,
                                    load_file ,
                                    data_size);

    forward_load_context_add_message(load_context, msg);

    free( msg );
    free( load_file );
  }
}


static void enkf_state_log_custom_kw_load(const enkf_node_type * enkf_node, int report_step, forward_load_context_type * load_context) {
  if (forward_load_context_accept_messages(load_context)) {
    char * load_file = enkf_config_node_alloc_infile(enkf_node_get_config(enkf_node), report_step);
    char * msg = util_alloc_sprintf("Loaded CUSTOM_KW: %s instance for step: %d from file: %s",
                                    enkf_node_get_key(enkf_node),
                                    report_step,
                                    load_file);

    forward_load_context_add_message(load_context, msg);

    free(msg);
    free(load_file);
  }
}



static int_vector_type * __enkf_state_get_time_index(enkf_fs_type * sim_fs, const ecl_sum_type * summary) {
  time_map_type * time_map = enkf_fs_get_time_map( sim_fs );
  time_map_summary_update( time_map , summary );
  return time_map_alloc_index_map( time_map , summary );
}


/*
 * Check if there are summary keys in the ensemble config that is not found in Eclipse. If this is the case, AND we
 * have observations for this key, we have a problem. Otherwise, just print a message to the log.
 */
static void enkf_state_check_for_missing_eclipse_summary_data(const ensemble_config_type * ens_config,
                                                              const summary_key_matcher_type * matcher,
                                                              const ecl_smspec_type * smspec,
                                                              forward_load_context_type * load_context,
                                                              const int iens ) {

  stringlist_type * keys = summary_key_matcher_get_keys(matcher);

  for (int i = 0; i < stringlist_get_size(keys); i++) {

    const char *key = stringlist_iget(keys, i);

    if (ecl_smspec_has_general_var(smspec, key) || !summary_key_matcher_summary_key_is_required(matcher, key))
      continue;

    if (!ensemble_config_has_key(ens_config, key))
      continue;

    const enkf_config_node_type *config_node = ensemble_config_get_node(ens_config, key);
    if (enkf_config_node_get_num_obs(config_node) == 0) {
      res_log_finfo("[%03d:----] Unable to find Eclipse data for summary key: "
                    "%s, but have no observations either, so will continue.",
                    iens, key);
    } else {
      res_log_ferror("[%03d:----] Unable to find Eclipse data for summary key: "
                     "%s, but have observation for this, job will fail.",
                     iens, key);
      forward_load_context_update_result(load_context, LOAD_FAILURE);
      if (forward_load_context_accept_messages(load_context)) {
        char *msg = util_alloc_sprintf("Failed to load vector: %s", key);
        forward_load_context_add_message(load_context, msg);
        free(msg);
      }
    }
  }

  stringlist_free(keys);
}

static bool enkf_state_internalize_dynamic_eclipse_results(ensemble_config_type * ens_config,
                                                           forward_load_context_type * load_context ,
                                                           const model_config_type * model_config) {

  bool load_summary = ensemble_config_has_impl_type(ens_config, SUMMARY);
  const run_arg_type * run_arg = forward_load_context_get_run_arg( load_context );
  const summary_key_matcher_type * matcher = ensemble_config_get_summary_key_matcher(ens_config);
  const ecl_sum_type * summary = forward_load_context_get_ecl_sum( load_context );
  int matcher_size = summary_key_matcher_get_size(matcher);

  if (load_summary || matcher_size > 0 || summary) {
    int load_start = run_arg_get_load_start( run_arg );

    if (load_start == 0) { /* Do not attempt to load the "S0000" summary results. */
      load_start++;
    }

    {
      enkf_fs_type * sim_fs = run_arg_get_sim_fs( run_arg );
      /** OK - now we have actually loaded the ecl_sum instance, or ecl_sum == NULL. */
      if (summary) {
        int_vector_type * time_index = __enkf_state_get_time_index(sim_fs, summary);

        /*
          Now there are two related / conflicting(?) systems for
          checking summary time consistency, both internally in the
          time_map and also through the
          model_config_report_step_compatible() function.
        */

        /*Check the loaded summary against the reference ecl_sum_type */
        if (!model_config_report_step_compatible(model_config, summary))
          forward_load_context_update_result(load_context, REPORT_STEP_INCOMPATIBLE);


        /* The actual loading internalizing - from ecl_sum -> enkf_node. */
        const int iens   = run_arg_get_iens( run_arg );
        const int step2  = ecl_sum_get_last_report_step( summary );  /* Step2 is just taken from the number of steps found in the summary file. */

        int_vector_iset_block( time_index , 0 , load_start , -1 );
        int_vector_resize( time_index , step2 + 1, -1);

        const ecl_smspec_type * smspec = ecl_sum_get_smspec(summary);

        for(int i = 0; i < ecl_smspec_num_nodes(smspec); i++) {
          const ecl::smspec_node& smspec_node = ecl_smspec_iget_node_w_node_index(smspec, i);
          const char * key = smspec_node.get_gen_key1();

          if(summary_key_matcher_match_summary_key(matcher, key)) {
            summary_key_set_type * key_set = enkf_fs_get_summary_key_set(sim_fs);
            summary_key_set_add_summary_key(key_set, key);

            enkf_config_node_type * config_node = ensemble_config_get_or_create_summary_node(ens_config, key);
            enkf_node_type * node = enkf_node_alloc( config_node );

            enkf_node_try_load_vector( node , sim_fs , iens );  // Ensure that what is currently on file is loaded before we update.

            enkf_node_forward_load_vector( node , load_context , time_index);
            enkf_node_store_vector( node , sim_fs , iens );
            enkf_node_free( node );
          }
        }

        int_vector_free( time_index );

        /*
          Check if some of the specified keys are missing from the Eclipse data, and if there are observations for them. That is a problem.
        */
        enkf_state_check_for_missing_eclipse_summary_data(ens_config, matcher, smspec, load_context, iens);

        return true;
      } else {
        res_log_fwarning("Could not load ECLIPSE summary data from %s - "
                         "this will probably fail later ... ",
                         run_arg_get_runpath(run_arg));
        return false;
      }
    }
  } else {
    return true;
  }
}






static void enkf_state_internalize_custom_kw(const ensemble_config_type * ens_config,
                                             forward_load_context_type * load_context ,
                                             const model_config_type * model_config) {

  const run_arg_type * run_arg     = forward_load_context_get_run_arg( load_context );
  enkf_fs_type *sim_fs             = run_arg_get_sim_fs(run_arg);
  const int iens                   = run_arg_get_iens( run_arg );
  stringlist_type * custom_kw_keys = ensemble_config_alloc_keylist_from_impl_type(ens_config, CUSTOM_KW);
  const int report_step            = 0;

  custom_kw_config_set_type * config_set = enkf_fs_get_custom_kw_config_set(sim_fs);
  custom_kw_config_set_reset(config_set);

  for (int ikey=0; ikey < stringlist_get_size(custom_kw_keys); ikey++) {
    const char* custom_kw_key = stringlist_iget(custom_kw_keys, ikey);
    enkf_config_node_type * config_node = ensemble_config_get_node( ens_config , custom_kw_key);
    enkf_node_type * node = enkf_node_alloc( config_node );

    if (enkf_node_vector_storage(node))
      util_abort("%s: Vector storage not correctly implemented for CUSTOM_KW\n", __func__);

    if (enkf_node_internalize(node, report_step) && enkf_node_has_func(node, forward_load_func)) {
      if (enkf_node_forward_load(node, load_context)) {
        node_id_type node_id = {.report_step = report_step, .iens = iens };

        enkf_node_store(node, sim_fs, false, node_id);

        const enkf_config_node_type * config_node = enkf_node_get_config(node);
        const custom_kw_config_type * custom_kw_config = (const custom_kw_config_type *)(custom_kw_config_type*) enkf_config_node_get_ref(config_node);
        custom_kw_config_set_add_config(config_set, custom_kw_config);
        enkf_state_log_custom_kw_load(node, report_step, load_context);
      } else {
        forward_load_context_update_result(load_context, LOAD_FAILURE);
        res_log_ferror("[%03d:%04d] Failed load data for CUSTOM_KW node: %s.",
                       iens, report_step, enkf_node_get_key(node));

        if (forward_load_context_accept_messages(load_context)) {
          char * msg = util_alloc_sprintf("Failed to load: %s at step: %d", enkf_node_get_key(node), report_step);
          forward_load_context_add_message(load_context , msg);
          free( msg );
        }
      }
    }
    enkf_node_free( node );
  }

  stringlist_free(custom_kw_keys);
}


static void enkf_state_load_gen_data_node(
  forward_load_context_type * load_context,
  enkf_fs_type * sim_fs,
  int iens,
  const enkf_config_node_type * config_node,
  int start,
  int stop)
{
  for (int report_step = start; report_step <= stop; report_step++) {
    if (!enkf_config_node_internalize(config_node, report_step))
      continue;

    forward_load_context_select_step(load_context, report_step);
    enkf_node_type * node = enkf_node_alloc(config_node);

    if (enkf_node_forward_load(node, load_context)) {
      node_id_type node_id = {.report_step = report_step,
                              .iens = iens };

      enkf_node_store(node, sim_fs, false, node_id);
      enkf_state_log_GEN_DATA_load(node, report_step, load_context);
    } else {
      forward_load_context_update_result(load_context, LOAD_FAILURE);
      res_log_ferror("[%03d:%04d] Failed load data for GEN_DATA node:%s.",
                     iens, report_step, enkf_node_get_key(node));

      if (forward_load_context_accept_messages(load_context)) {
        char * msg = util_alloc_sprintf("Failed to load: %s at step:%d",
                                        enkf_node_get_key(node), report_step);
        forward_load_context_add_message(load_context, msg);
        free(msg);
      }
    }
    enkf_node_free(node);
  }
}


static void enkf_state_internalize_GEN_DATA(const ensemble_config_type * ens_config,
                                            forward_load_context_type * load_context ,
                                            const model_config_type * model_config ,
                                            int last_report) {

  stringlist_type * keylist_GEN_DATA = ensemble_config_alloc_keylist_from_impl_type(ens_config, GEN_DATA);

  int numkeys = stringlist_get_size(keylist_GEN_DATA);

  if (numkeys > 0)
    if (last_report <= 0)
      res_log_fwarning("Trying to load GEN_DATA without properly "
                       "set last_report (was %d) - will only look for step 0 data: %s",
                       last_report,
                       stringlist_iget(keylist_GEN_DATA, 0)
        );

  const run_arg_type * run_arg = forward_load_context_get_run_arg(load_context);
  enkf_fs_type * sim_fs        = run_arg_get_sim_fs(run_arg);
  const int iens               = run_arg_get_iens(run_arg);

  for (int ikey=0; ikey < numkeys; ikey++) {
    const enkf_config_node_type * config_node = ensemble_config_get_node(ens_config,
                                                                         stringlist_iget(keylist_GEN_DATA,
                                                                                         ikey));

    /*
      This for loop should probably be changed to use the report
      steps configured in the gen_data_config object, instead of
      spinning through them all.
    */
    int start = run_arg_get_load_start(run_arg);
    int stop  = util_int_max(0, last_report);  // inclusive
    enkf_state_load_gen_data_node(load_context,
                                  sim_fs,
                                  iens,
                                  config_node,
                                  start,
                                  stop);
  }
  stringlist_free( keylist_GEN_DATA );
}





static forward_load_context_type * enkf_state_alloc_load_context(const ensemble_config_type * ens_config,
                                                                 const ecl_config_type * ecl_config,
                                                                 const run_arg_type * run_arg,
                                                                 stringlist_type * messages) {
  bool load_summary = false;
  const summary_key_matcher_type * matcher = ensemble_config_get_summary_key_matcher(ens_config);
  if (summary_key_matcher_get_size(matcher) > 0)
    load_summary = true;

  if (ensemble_config_require_summary(ens_config))
    load_summary = true;

  forward_load_context_type * load_context;

  load_context = forward_load_context_alloc(run_arg,
                                            load_summary,
                                            ecl_config,
                                            messages);
  return load_context;

}


/**
   This function loads the results from a forward simulations from report_step1
   to report_step2. The details of what to load are in model_config and the
   spesific nodes for special cases.

   Will mainly be called at the end of the forward model, but can also
   be called manually from external scope.
*/
static int enkf_state_internalize_results(ensemble_config_type * ens_config,
                                          model_config_type * model_config,
                                          const ecl_config_type * ecl_config,
                                          const run_arg_type * run_arg,
                                          stringlist_type * msg_list) {

  forward_load_context_type * load_context = enkf_state_alloc_load_context( ens_config, ecl_config, run_arg, msg_list);
  /*
    The timing information - i.e. mainly what is the last report step
    in these results are inferred from the loading of summary results,
    hence we must load the summary results first.
  */

  enkf_state_internalize_dynamic_eclipse_results(ens_config,
                                                 load_context ,
                                                 model_config);

  enkf_fs_type * sim_fs = run_arg_get_sim_fs( run_arg );
  int last_report = time_map_get_last_step( enkf_fs_get_time_map( sim_fs ));
  if (last_report < 0)
    last_report = model_config_get_last_history_restart( model_config );

  /* Ensure that the last step is internalized? */
  if (last_report > 0)
    model_config_set_internalize_state( model_config , last_report);

  enkf_state_internalize_GEN_DATA(ens_config , load_context , model_config , last_report);
  enkf_state_internalize_custom_kw(ens_config, load_context , model_config);

  int result = forward_load_context_get_result(load_context);
  forward_load_context_free( load_context );
  return result;
}





static int enkf_state_load_from_forward_model__(ensemble_config_type * ens_config,
                                                model_config_type * model_config,
                                                const ecl_config_type * ecl_config,
                                                const run_arg_type * run_arg ,
                                                stringlist_type * msg_list) {

  int result = 0;

  if (ensemble_config_have_forward_init( ens_config ))
    result |= ensemble_config_forward_init( ens_config , run_arg );

  result |= enkf_state_internalize_results( ens_config, model_config, ecl_config, run_arg , msg_list );
  state_map_type * state_map = enkf_fs_get_state_map( run_arg_get_sim_fs( run_arg ) );
  int iens = run_arg_get_iens( run_arg );
  if (result & LOAD_FAILURE)
    state_map_iset( state_map , iens , STATE_LOAD_FAILURE);
  else
    state_map_iset( state_map , iens , STATE_HAS_DATA);

  return result;
}

int enkf_state_load_from_forward_model(enkf_state_type * enkf_state ,
                                       run_arg_type * run_arg ,
                                       stringlist_type * msg_list) {

  ensemble_config_type * ens_config = enkf_state->ensemble_config;
  model_config_type * model_config = enkf_state->shared_info->model_config;
  const ecl_config_type * ecl_config = enkf_state->shared_info->ecl_config;

  return enkf_state_load_from_forward_model__( ens_config, model_config, ecl_config, run_arg, msg_list);
}


/**
   Observe that this does not return the loadOK flag; it will load as
   good as it can all the data it should, and be done with it.
*/

void * enkf_state_load_from_forward_model_mt( void * arg ) {
  arg_pack_type * arg_pack     = arg_pack_safe_cast( arg );
  enkf_state_type * enkf_state = enkf_state_safe_cast((enkf_state_type * ) arg_pack_iget_ptr( arg_pack  , 0 ));
  run_arg_type * run_arg       = (run_arg_type * ) arg_pack_iget_ptr( arg_pack  , 1 );
  stringlist_type * msg_list   = (stringlist_type * ) arg_pack_iget_ptr( arg_pack  , 2 );
  bool manual_load             = arg_pack_iget_bool( arg_pack , 3 );
  int * result                 = (int * ) arg_pack_iget_ptr( arg_pack  , 4 );
  int iens                     = run_arg_get_iens( run_arg );

  if (manual_load)
    state_map_update_undefined(enkf_fs_get_state_map( run_arg_get_sim_fs(run_arg) ) , iens , STATE_INITIALIZED);

  *result = enkf_state_load_from_forward_model( enkf_state , run_arg , msg_list );
  if (*result & REPORT_STEP_INCOMPATIBLE) {
    // If refcase has been used for observations: crash and burn.
    fprintf(stderr,"** Warning the timesteps in refcase and current simulation are not in accordance - something wrong with schedule file?\n");
    *result -= REPORT_STEP_INCOMPATIBLE;
  }

  return NULL;
}





void enkf_state_free(enkf_state_type *enkf_state) {
  hash_free(enkf_state->node_hash);
  shared_info_free(enkf_state->shared_info);
  free(enkf_state);
}







/**
   init_step    : The parameters are loaded from this EnKF/report step.
   report_step1 : The simulation should start from this report step;
                  dynamic data are loaded from this step.
   report_step2 : The simulation should stop at this report step. (unless run_mode == ENSEMBLE_PREDICTION - where it just runs til end.)

   For a normal EnKF run we well have init_step == report_step1, but
   in the case where we want rerun from the beginning with updated
   parameters, they will be different. If init_step != report_step1,
   it is required that report_step1 == 0; otherwise the dynamic data
   will become completely inconsistent. We just don't allow that!
*/

void enkf_state_init_eclipse(const res_config_type * res_config,
                             const run_arg_type * run_arg ) {

  ensemble_config_type * ens_config = res_config_get_ensemble_config(res_config);
  const ecl_config_type * ecl_config = res_config_get_ecl_config(res_config);
  model_config_type * model_config = res_config_get_model_config(res_config);

  util_make_path(run_arg_get_runpath(run_arg));

  ert_templates_instansiate(res_config_get_templates(res_config),
                            run_arg_get_runpath(run_arg),
                            run_arg_get_subst_list(run_arg));

  enkf_state_ecl_write(ens_config,
                       model_config,
                       run_arg,
                       run_arg_get_sim_fs(run_arg));

  /* Writing the ECLIPSE data file. */
  if (ecl_config_have_eclbase(ecl_config) && ecl_config_get_data_file(ecl_config)) {
    char * data_file = ecl_util_alloc_filename(run_arg_get_runpath(run_arg),
                                               run_arg_get_job_name(run_arg),
                                               ECL_DATA_FILE,
                                               true,
                                               -1);

    subst_list_update_string(run_arg_get_subst_list(run_arg), &data_file);
    subst_list_filter_file(run_arg_get_subst_list(run_arg),
                           ecl_config_get_data_file(ecl_config),
                           data_file);

    free(data_file);
  }

  mode_t umask = site_config_get_umask(res_config_get_site_config( res_config ));

  /* This is where the job script is created */
  const env_varlist_type * varlist = site_config_get_env_varlist(res_config_get_site_config(res_config));
  forward_model_formatted_fprintf(model_config_get_forward_model(model_config),
                                  run_arg_get_run_id( run_arg ),
                                  run_arg_get_runpath(run_arg),
                                  model_config_get_data_root(model_config),
                                  run_arg_get_subst_list(run_arg),
                                  umask,
                                  varlist);
}



/**
    Observe that if run_arg == false, this routine will return with
    job_completeOK == true, that might be a bit misleading.

    Observe that if an internal retry is performed, this function will
    be called several times - MUST BE REENTRANT.
*/

bool enkf_state_complete_forward_modelOK(const res_config_type * res_config,
                                                run_arg_type * run_arg) {

  ensemble_config_type * ens_config = res_config_get_ensemble_config( res_config );
  const ecl_config_type * ecl_config = res_config_get_ecl_config( res_config );
  model_config_type * model_config = res_config_get_model_config( res_config );
  const int iens = run_arg_get_iens( run_arg );
  int result;


  /**
     The queue system has reported that the run is OK, i.e. it has
     completed and produced the targetfile it should. We then check
     in this scope whether the results can be loaded back; if that
     is OK the final status is updated, otherwise: restart.
  */
  res_log_finfo("[%03d:%04d-%04d] Forward model complete - starting to load results.",
                iens, run_arg_get_step1(run_arg), run_arg_get_step2(run_arg));

  result = enkf_state_load_from_forward_model__( ens_config,
                                                 model_config,
                                                 ecl_config,
                                                 run_arg,
                                                 NULL);

  if (result & REPORT_STEP_INCOMPATIBLE) {
    // If refcase has been used for observations: crash and burn.
     fprintf(stderr,"** Warning the timesteps in refcase and current simulation are not in accordance - something wrong with schedule file?\n");
     result -= REPORT_STEP_INCOMPATIBLE;
  }


  if (result == 0) {
    /*
      The loading succeded - so this is a howling success! We set
      the main status to JOB_QUEUE_ALL_OK and inform the queue layer
      about the success. In addition we set the simple status
      (should be avoided) to JOB_RUN_OK.
    */
    run_arg_set_run_status( run_arg , JOB_RUN_OK);
    res_log_finfo("[%03d:%04d-%04d] Results loaded successfully.",
                  iens, run_arg_get_step1(run_arg), run_arg_get_step2(run_arg));

  }

  return (result == 0) ? true : false;
}


bool enkf_state_complete_forward_modelOK__(void * arg ) {
  callback_arg_type * cb_arg = callback_arg_safe_cast( arg );

  return enkf_state_complete_forward_modelOK( cb_arg->res_config,
                                              cb_arg->run_arg );
}



bool enkf_state_complete_forward_model_EXIT_handler__(run_arg_type * run_arg) {
  const int iens = run_arg_get_iens( run_arg );
  res_log_ferror("[%03d:%04d-%04d] FAILED COMPLETELY.",
                 iens, run_arg_get_step1(run_arg), run_arg_get_step2(run_arg));

  if (run_arg_get_run_status(run_arg) != JOB_LOAD_FAILURE)
    run_arg_set_run_status( run_arg , JOB_RUN_FAILURE);

  state_map_type * state_map = enkf_fs_get_state_map(run_arg_get_sim_fs( run_arg ));
  state_map_iset(state_map, iens, STATE_LOAD_FAILURE);
  return false;
}


static bool enkf_state_complete_forward_model_EXIT_handler(void * arg) {
  callback_arg_type * callback_arg = callback_arg_safe_cast( arg );
  run_arg_type * run_arg = callback_arg->run_arg;
  return enkf_state_complete_forward_model_EXIT_handler__( run_arg);
}


bool enkf_state_complete_forward_modelEXIT__(void * arg ) {
  return enkf_state_complete_forward_model_EXIT_handler(arg);
}


/**
    This function is called when:

     1. The external queue system has said that everything is OK; BUT
        the ert layer failed to load all the data.

     2. The external queue system has seen the job fail.

    The parameter and state variables will be resampled before
    retrying. And all random elements in templates+++ will be
    resampled.
*/



static void enkf_state_internal_retry(const res_config_type * res_config,
                                      run_arg_type * run_arg,
                                      rng_type * rng) {
  ensemble_config_type * ens_config = res_config_get_ensemble_config( res_config );
  const int iens = run_arg_get_iens( run_arg );

  res_log_ferror("[%03d:%04d - %04d] Forward model failed.",
                 iens, run_arg_get_step1(run_arg), run_arg_get_step2(run_arg));
  if (run_arg_can_retry( run_arg ) ) {
    res_log_ferror("[%03d] Resampling and resubmitting realization.", iens);

    stringlist_type * init_keys = ensemble_config_alloc_keylist_from_var_type( ens_config , PARAMETER );
    for (int ikey=0; ikey < stringlist_get_size( init_keys ); ikey++) {
      const enkf_config_node_type * config_node = ensemble_config_get_node( ens_config , stringlist_iget( init_keys , ikey) );
      enkf_node_type * node = enkf_node_alloc( config_node );
      if (enkf_node_initialize( node , iens , rng )) {
        node_id_type node_id = { .report_step = 0, .iens = iens };
        enkf_node_store(node, run_arg_get_sim_fs( run_arg ), true, node_id);
      }
      enkf_node_free( node );
    }
    stringlist_free( init_keys );

    /* Possibly clear the directory and do a FULL rewrite of ALL the necessary files. */
    enkf_state_init_eclipse(res_config, run_arg);
    run_arg_increase_submit_count( run_arg );
  }
}


bool enkf_state_complete_forward_modelRETRY__(void * arg ) {
  callback_arg_type * cb_arg = callback_arg_safe_cast( arg );

  if (run_arg_can_retry(cb_arg->run_arg)) {
    enkf_state_internal_retry( cb_arg->res_config,
                               cb_arg->run_arg,
                               cb_arg->rng);
    return true;
  }

  return false;
}



/*****************************************************************/





const ensemble_config_type * enkf_state_get_ensemble_config( const enkf_state_type * enkf_state ) {
  return enkf_state->ensemble_config;
}


/**
  This function writes out all the files needed by an ECLIPSE simulation, this
  includes the restart file, and the various INCLUDE files corresponding to
  parameters estimated by EnKF.

  The writing of restart file is delegated to enkf_state_write_restart_file().
*/

// TODO: enkf_fs_type could be fetched from run_arg
void enkf_state_ecl_write(const ensemble_config_type * ens_config, const model_config_type * model_config, const run_arg_type * run_arg , enkf_fs_type * fs) {
  /**
     This iteration manipulates the hash (thorugh the enkf_state_del_node() call)

     -----------------------------------------------------------------------------------------
     T H I S  W I L L  D E A D L O C K  I F  T H E   H A S H _ I T E R  A P I   I S   U S E D.
     -----------------------------------------------------------------------------------------
  */
  int iens                         = run_arg_get_iens( run_arg );
  const char * base_name           = model_config_get_gen_kw_export_name(model_config);
  value_export_type * export_value = value_export_alloc( run_arg_get_runpath( run_arg ), base_name );

  stringlist_type * key_list = ensemble_config_alloc_keylist_from_var_type( ens_config , PARAMETER + EXT_PARAMETER);
  for (int ikey = 0; ikey < stringlist_get_size( key_list ); ikey++) {
    enkf_config_node_type * config_node = ensemble_config_get_node( ens_config, stringlist_iget( key_list , ikey));
    enkf_node_type * enkf_node = enkf_node_alloc( config_node );
    bool forward_init = enkf_node_use_forward_init( enkf_node );
    node_id_type node_id = {.report_step = run_arg_get_step1(run_arg),
                            .iens = iens };

    if ((run_arg_get_step1(run_arg) == 0) && (forward_init)) {

      if (enkf_node_has_data( enkf_node , fs , node_id))
        enkf_node_load(enkf_node, fs, node_id);
      else
        continue;
    } else
      enkf_node_load(enkf_node, fs, node_id);

    enkf_node_ecl_write(enkf_node , run_arg_get_runpath( run_arg ) , export_value , run_arg_get_step1(run_arg));
    enkf_node_free(enkf_node);
  }
  value_export( export_value );

  value_export_free( export_value );
  stringlist_free( key_list );
}


#include "enkf_state_nodes.cpp"
