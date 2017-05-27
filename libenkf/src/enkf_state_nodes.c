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

    /* Setting the global subst list so that the GEN_KW templates can contain e.g. <IENS> and <CWD>. */
    if (enkf_node_get_impl_type( enkf_node ) == GEN_KW)
      gen_kw_set_subst_parent( enkf_node_value_ptr( enkf_node ) , enkf_state->subst_list );
  }
}


static enkf_node_type * enkf_state_get_or_create_node(enkf_state_type * enkf_state, const enkf_config_node_type * config_node) {
    const char * key = enkf_config_node_get_key(config_node);
    if(!enkf_state_has_node(enkf_state, key)) {
        enkf_state_add_node(enkf_state, key, config_node);
    }
    return enkf_state_get_node(enkf_state, key);
}

/**
   This function loads the STATE from a forward simulation. In ECLIPSE
   speak that means to load the solution vectors (PRESSURE/SWAT/..)
   and the necessary static keywords.

   When the state has been loaded it goes straight to disk.
*/

static void enkf_state_internalize_eclipse_state(enkf_state_type * enkf_state ,
						 forward_load_context_type * load_context,
						 const model_config_type * model_config ,
                                                 int report_step ,
                                                 bool store_vectors) {

  shared_info_type   * shared_info   = enkf_state->shared_info;
  const ecl_config_type * ecl_config = shared_info->ecl_config;
  if (!ecl_config_active( ecl_config ))
    return;

  if (!model_config_internalize_state( model_config , report_step ))
    return;

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
    enkf_fs_type * result_fs = run_arg_get_result_fs( run_arg );

    member_config_type * my_config     = enkf_state->my_config;
    const int  iens                    = member_config_get_iens( my_config );
    const bool internalize_state       = model_config_internalize_state( model_config , report_step );

    hash_iter_type * iter = hash_iter_alloc(enkf_state->node_hash);
    while ( !hash_iter_is_complete(iter) ) {
      enkf_node_type * enkf_node = hash_iter_get_next_value(iter);
      if (enkf_node_get_var_type(enkf_node) != DYNAMIC_STATE)
        continue;

      if (enkf_node_get_impl_type(enkf_node) != FIELD)
        continue;

      if (!enkf_node_has_func(enkf_node , forward_load_func))
        continue;

      if (internalize_state || enkf_node_internalize(enkf_node , report_step)) {

        if (enkf_node_forward_load(enkf_node , load_context)) {
          node_id_type node_id = {.report_step = report_step ,
                                  .iens = iens };
          enkf_node_store( enkf_node , result_fs, store_vectors , node_id );
        } else {
          forward_load_context_update_result(load_context, LOAD_FAILURE);
          res_log_add_fmt_message( 1 , NULL , "[%03d:%04d] Failed load data for FIELD node:%s.",iens , report_step , enkf_node_get_key( enkf_node ));

          if (forward_load_context_accept_messages(load_context)) {
            char * msg = util_alloc_sprintf("Failed to load: %s at step:%d" , enkf_node_get_key( enkf_node ) , report_step);
            forward_load_context_add_message(load_context, msg);
            free( msg );
          }
        }

      }
    }
    hash_iter_free(iter);
  }
}


int enkf_state_forward_init(enkf_state_type * enkf_state ,
                            run_arg_type * run_arg) {

  int result = 0;
  if (run_arg_get_step1(run_arg) == 0) {
    int iens = enkf_state_get_iens( enkf_state );
    hash_iter_type * iter = hash_iter_alloc( enkf_state->node_hash );
    while ( !hash_iter_is_complete(iter) ) {
      enkf_node_type * node = hash_iter_get_next_value(iter);
      if (enkf_node_use_forward_init(node)) {
        enkf_fs_type * result_fs = run_arg_get_result_fs( run_arg );
        node_id_type node_id = {.report_step = 0 ,
                                .iens = iens };


        /*
           Will not reinitialize; i.e. it is essential that the
           forward model uses the state given from the stored
           instance, and not from the current run of e.g. RMS.
        */

        if (!enkf_node_has_data( node , result_fs , node_id)) {
          if (enkf_node_forward_init(node , run_arg_get_runpath( run_arg ) , iens ))
            enkf_node_store( node , result_fs , false , node_id );
          else {
            char * init_file = enkf_config_node_alloc_initfile( enkf_node_get_config( node ) , run_arg_get_runpath(run_arg) , iens );

            if (init_file && !util_file_exists( init_file ))
              fprintf(stderr,"File not found: %s - failed to initialize node: %s\n", init_file , enkf_node_get_key( node ));
            else
              fprintf(stderr,"Failed to initialize node: %s\n", enkf_node_get_key( node ));

            util_safe_free( init_file );
            result |= LOAD_FAILURE;
          }
        }

      }
    }
    hash_iter_free( iter );
  }
  return result;
}





static void enkf_state_fread(enkf_state_type * enkf_state , enkf_fs_type * fs , int mask , int report_step ) {
  const member_config_type * my_config = enkf_state->my_config;
  const int num_keys = hash_get_size(enkf_state->node_hash);
  char ** key_list   = hash_alloc_keylist(enkf_state->node_hash);
  int ikey;

  for (ikey = 0; ikey < num_keys; ikey++) {
    enkf_node_type * enkf_node = hash_get(enkf_state->node_hash , key_list[ikey]);
    if (enkf_node_include_type(enkf_node , mask)) {
      node_id_type node_id = {.report_step = report_step ,
                              .iens = member_config_get_iens( my_config )};
      bool forward_init = enkf_node_use_forward_init( enkf_node );
      if (forward_init)
        enkf_node_try_load(enkf_node , fs , node_id );
      else
        enkf_node_load(enkf_node , fs , node_id);
    }
  }
  util_free_stringlist(key_list , num_keys);
}


/**
   This function will load all the nodes listed in the current
   restart_kw_list; in addition to all other variable of type
   DYNAMIC_STATE. Observe that for DYNAMIC state nodes it will try
   firt analyzed state and then forecast state.
*/


static void enkf_state_fread_state_nodes(enkf_state_type * enkf_state , enkf_fs_type * fs , int report_step ) {
  const member_config_type * my_config = enkf_state->my_config;
  const int iens                       = member_config_get_iens( my_config );

  /*
     First pass - load all the STATIC nodes. It is essential to use
     the restart_kw_list when loading static nodes, otherwise static
     nodes which were only present at e.g. step == 0 will create
     problems: (They are in the enkf_state hash table because they
     were seen at step == 0, but have not been seen subesquently and
     the loading fails.)
  */

  /* Second pass - DYNAMIC state nodes. */
  {
    const int num_keys = hash_get_size(enkf_state->node_hash);
    char ** key_list   = hash_alloc_keylist(enkf_state->node_hash);
    int ikey;

    for (ikey = 0; ikey < num_keys; ikey++) {
      enkf_node_type * enkf_node = hash_get(enkf_state->node_hash , key_list[ikey]);
      enkf_var_type var_type = enkf_node_get_var_type( enkf_node );
      node_id_type node_id = {.report_step = report_step ,
                              .iens = iens };

      if (var_type == DYNAMIC_STATE) {
        /*
           Here the enkf_node_try_load() function is used NOT because we accept
           that the node is not present, but because the try_fread()
           function accepts the BOTH state type.
        */
        if (!enkf_node_try_load(enkf_node , fs , node_id))
          util_abort("%s: failed to load node:%s  report_step:%d iens:%d \n",__func__ , key_list[ikey] , report_step , iens  );
      }
    }
    util_free_stringlist(key_list , num_keys);
  }
}

/**
   This is a special function which is only used to load the initial
   state of dynamic_state nodes. It checks if the enkf_config_node has
   set a valid value for input_file, in that case that means we should
   also have an internalized representation of it, otherwise it will
   just return (i.e. for PRESSURE / SWAT).
*/

static void enkf_state_fread_initial_state(enkf_state_type * enkf_state , enkf_fs_type * fs) {
  const member_config_type * my_config = enkf_state->my_config;
  const int num_keys = hash_get_size(enkf_state->node_hash);
  char ** key_list   = hash_alloc_keylist(enkf_state->node_hash);
  int ikey;

  for (ikey = 0; ikey < num_keys; ikey++) {
    enkf_node_type * enkf_node = hash_get(enkf_state->node_hash , key_list[ikey]);
    if (enkf_node_get_var_type(enkf_node) == DYNAMIC_STATE) {
      const enkf_config_node_type * config_node = enkf_node_get_config( enkf_node );

      /* Just checked for != NULL */
      char * load_file = enkf_config_node_alloc_infile( config_node , 0);
      if (load_file != NULL) {
        node_id_type node_id = {.report_step = 0 ,
                                .iens  = member_config_get_iens( my_config ) };
        enkf_node_load(enkf_node , fs , node_id);
      }

      util_safe_free( load_file );
    }
  }
  util_free_stringlist(key_list , num_keys);
}



static enkf_node_type * enkf_state_get_node(const enkf_state_type * enkf_state , const char * node_key) {
  if (hash_has_key(enkf_state->node_hash , node_key)) {
    enkf_node_type * enkf_node = hash_get(enkf_state->node_hash , node_key);
    return enkf_node;
  } else {
    util_abort("%s: node:[%s] not found in state object - aborting.\n",__func__ , node_key);
    return NULL; /* Compiler shut up */
  }
}



