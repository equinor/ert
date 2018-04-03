static bool enkf_state_has_node(const enkf_state_type * enkf_state , const char * node_key) {
  bool has_node = hash_has_key(enkf_state->node_hash , node_key);
  return has_node;
}


static void enkf_state_del_node(enkf_state_type * enkf_state , const char * node_key) {
  if (hash_has_key(enkf_state->node_hash , node_key))
    hash_del(enkf_state->node_hash , node_key);
  else
    fprintf(stderr,"%s: tried to remove node:%s which is not in state - internal error?? \n",__func__ , node_key);
}



/**
   The enkf_state inserts a reference to the node object. The
   enkf_state object takes ownership of the node object, i.e. it will
   free it when the game is over.

   Observe that if the node already exists the existing node will be
   removed (freed and so on ... ) from the enkf_state object before
   adding the new; this was previously considered a run-time error.
*/


void enkf_state_add_node(enkf_state_type * enkf_state , const char * node_key , const enkf_config_node_type * config) {
  if (enkf_state_has_node(enkf_state , node_key))
    enkf_state_del_node( enkf_state , node_key );   /* Deleting the old instance (if we had one). */
  {
    enkf_node_type *enkf_node;
    if (enkf_config_node_get_impl_type( config ) == CONTAINER)
      enkf_node = enkf_node_alloc_shared_container( config , enkf_state->node_hash );
    else
      enkf_node = enkf_node_alloc( config );

    hash_insert_hash_owned_ref(enkf_state->node_hash , node_key , enkf_node, enkf_node_free__);
  }
}



/**
   This function loads the STATE from a forward simulation. In ECLIPSE
   speak that means to load the solution vectors (PRESSURE/SWAT/..)
   and the necessary static keywords.

   When the state has been loaded it goes straight to disk.
*/

static void enkf_state_internalize_eclipse_state(const ensemble_config_type * ens_config,
						 forward_load_context_type * load_context,
                                                 const run_arg_type * run_arg,
                                                 int report_step,
                                                 bool store_vectors) {

  forward_load_context_load_restart_file( load_context , report_step);

  /******************************************************************/
  /**
     Starting on the enkf_node_forward_load() function calls. This
     is where the actual loading is done. Observe that this loading
     might involve other load functions than the ones used for
     loading PRESSURE++ from ECLIPSE restart files (e.g. for
     loading seismic results..)
  */

  {
    const run_arg_type * run_arg = forward_load_context_get_run_arg( load_context );
    enkf_fs_type * sim_fs = run_arg_get_sim_fs( run_arg );

    const int  iens                    = run_arg_get_iens( run_arg );
    hash_iter_type * iter = ensemble_config_alloc_hash_iter( ens_config );
    while ( !hash_iter_is_complete(iter) ) {
      enkf_config_node_type * config_node = hash_iter_get_next_value(iter);
      if (enkf_config_node_get_impl_type(config_node) != FIELD)
        continue;

      if (!enkf_config_node_use_forward_init(config_node))
        continue;

      enkf_node_type * enkf_node = enkf_node_alloc( config_node );
      if (enkf_node_forward_load(enkf_node , load_context)) {
        node_id_type node_id = {.report_step = report_step ,
                                .iens = iens };
        enkf_node_store( enkf_node , sim_fs, store_vectors , node_id );
      } else {
        forward_load_context_update_result(load_context, LOAD_FAILURE);
        res_log_ferror("[%03d:%04d] Failed load data for FIELD node:%s.",
                       iens, report_step, enkf_node_get_key(enkf_node));

        if (forward_load_context_accept_messages(load_context)) {
          char * msg = util_alloc_sprintf("Failed to load: %s at step:%d" , enkf_node_get_key( enkf_node ) , report_step);
          forward_load_context_add_message(load_context, msg);
          free( msg );
        }
      }
      enkf_node_free( enkf_node );

    }
    hash_iter_free(iter);
  }
}







