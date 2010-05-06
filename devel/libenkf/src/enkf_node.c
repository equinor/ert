#include <string.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <enkf_node.h>
#include <enkf_config_node.h>
#include <util.h>
#include <field.h>
#include <summary.h>
#include <ecl_static_kw.h>
#include <gen_kw.h>
#include <path_fmt.h>
#include <gen_data.h>
#include <enkf_serialize.h>
#include <buffer.h>
#include <msg.h>

/**
   A small illustration (says more than thousand words ...) of how the
   enkf_node, enkf_config_node, field[1] and field_config[1] objects
   are linked.


     ================
     |              |   o-----------
     |  ================           |                =====================
     |  |              |   o--------                |                   |
     |	|  ================        |------------->  |                   |
     |	|  |		  |        |                |  enkf_config_node |
     |	|  |		  |        |                |                   |
     ===|  |  enkf_node   |  o------                |                   |
      o	|  |		  |                         |                   |
      |	===|		  |                         =====================
      |	 o |		  |                                   o
      |	 | ================                                   |
      |  |        o                                           |
      |  \        |					      |
      |   \       | 					      |
      |    |      |					      |
      |    |  	  | 					      |
      |    |  	  |  					      |
      |    |      |   					      |
     \|/   |	  |    					      |
     ======|======|==                       		     \|/
     |    \|/     | |   o-----------
     |  ==========|=====           |                =====================
     |  |        \|/   |   o--------                |                   |
     |	|  ================        |------------->  |                   |
     |	|  |		  |        |                |  field_config     |
     |	|  |		  |        |                |                   |
     ===|  |  field       |  o------                |                   |
       	|  |		  |                         |                   |
     	===|		  |                         =====================
     	   |		  |
     	   ================


   To summarize in words:

   * The enkf_node object is an abstract object, which again contains
     a spesific enkf_object, like e.g. the field objects shown
     here. In general we have an ensemble of enkf_node objects.

   * The enkf_node objects contain a pointer to a enkf_config_node
     object.

   * The enkf_config_node object contains a pointer to the spesific
     config object, i.e. field_config in this case.

   * All the field objects contain a pointer to a field_config object.


   [1]: field is just an example, and could be replaced with any of
        the enkf object types.
*/

/*-----------------------------------------------------------------*/

/**
   A note on memory 
   ================ 

   The enkf_nodes can consume large amounts of memory, and for large
   models/ensembles we have a situation where not all the
   members/fields can be in memory simultanouesly - such low-memory
   situations are not really supported at the moment, but we have
   implemented some support for such problems:

    o All enkf objects should have a xxx_realloc_data() function. This
      function should be implemented in such a way that it is always
      safe to call, i.e. if the object already has allocated data the
      function should just return.

    o All enkf objects should implement a xxx_free_data()
      function. This function free the data of the object, and set the
      data pointer to NULL.


   The following 'rules' apply to the memory treatment:
   ----------------------------------------------------

    o Functions writing to memory can always be called, and it is their
      responsibility to allocate memory before actually writing on it. The
      writer functions are:

        enkf_node_initialize()
        enkf_node_fread()
        enkf_node_ecl_load()  

      These functions should all start with a call to
      enkf_node_ensure_memory(). The (re)allocation of data is done at
      the enkf_node level, and **NOT** in the low level object
      (altough that is where it is eventually done of course).

    o When it comes to functions reading memory it is a bit more
      tricky. It could be that if the functions are called without
      memory, that just means that the object is not active or
      something (and the function should just return). On the other
      hand trying to read a NULL pointer does indicate that program
      logic is not fully up to it? And should therefor maybe be
      punished?

    o The only memory operation which is exported to 'user-space'
      (i.e. the enkf_state object) is enkf_node_free_data(). 

*/

/**
   Keeeping track of node state.
   =============================

   To keep track of the state of the node's data (actually the data of
   the contained enkf_object, i.e. a field) we have three higly
   internal variables __state, __modified , __iens, and
   __report_step. These three variables are used/updated in the
   following manner:
   


    1. The nodes are created with (modified, report_step, state, iens) ==
       (true , -1 , undefined , -1).

    2. After initialization we set: report_step -> 0 , state ->
       analyzed, modified -> true, iens -> -1

    3. After load (both from ensemble and ECLIPSE). We set modified ->
       false, and report_step, state and iens according to the load
       arguments.
      
    4. After deserialize (i.e. update) we set modified -> true.
      
    5. After write (to ensemble) we set in the same way as after load.
  
    6. After free_data we invalidate according to the newly allocated
       status.

    7. In the ens_load routine we check if modified == false and the
       report_step and state arguments agree with the current
       values. IN THAT CASE WE JUST RETURN WITHOUT ACTUALLY HITTING
       THE FILESYSTEM. This performance gain is the main point of the
       whole excercise.
*/




struct enkf_node_struct {
  alloc_ftype                    * alloc;
  ecl_write_ftype                * ecl_write;
  ecl_load_ftype                 * ecl_load;
  free_data_ftype                * free_data;
  user_get_ftype                 * user_get;
  set_data_ftype                 * set_data;
  set_inflation_ftype            * set_inflation;
  fload_ftype                    * fload;

  matrix_serialize_ftype         * matrix_serialize;
  matrix_deserialize_ftype       * matrix_deserialize;
  load_ftype                     * load;
  store_ftype                    * store;    
  initialize_ftype   		 * initialize;
  node_free_ftype                * freef;
  clear_ftype        		 * clear;
  node_copy_ftype                * copy;
  scale_ftype        		 * scale;
  iadd_ftype         		 * iadd;
  imul_ftype         		 * imul;
  isqrt_ftype        		 * isqrt;
  iaddsqr_ftype      		 * iaddsqr;
  
  /******************************************************************/
  char               *node_key;       	    /* The (hash)key this node is identified with. */
  void               *data;                 /* A pointer to the underlying enkf_object, i.e. multflt_type instance, or a field_type instance or ... */
  const enkf_config_node_type *config;      /* A pointer to a enkf_config_node instance (which again cointans a pointer to the config object of data). */
  
  /*****************************************************************/
  /* The variables below this line are VERY INTERNAL.              */
  bool                __modified;           /* __modified, __report_step, __iens and __state are internal variables trying  */
  int                 __report_step;        /* to record the state of the in-memory reporesentation of the node->data. See */ 
  state_enum          __state;              /* the documentation with heading "Keeping track of node state". */
  int                 __iens;               /* Observe that this __iens  variable "should not be used" - a node can change __iens value during run. */
};





const enkf_config_node_type * enkf_node_get_config(const enkf_node_type * node) {
  return node->config;
}


/*****************************************************************/

/*
  All the function pointers REALLY should be in the config object ...
*/


#define FUNC_ASSERT(func) if (func == NULL) util_abort("%s: function handler: %s not registered for node:%s - aborting\n",__func__ , #func , enkf_node->node_key); 



//static void enkf_node_ensure_memory(enkf_node_type * enkf_node) {
//  FUNC_ASSERT(enkf_node->realloc_data);
//  if (!enkf_node->__memory_allocated) {
//    enkf_node->realloc_data(enkf_node->data);
//    enkf_node->__memory_allocated = true;
//  }
//}



void enkf_node_alloc_domain_object(enkf_node_type * node) {
  if (node->data != NULL)
    node->freef(node->data);
  node->data = node->alloc(enkf_config_node_get_ref(node->config));
}







enkf_node_type * enkf_node_copyc(const enkf_node_type * enkf_node) {
  FUNC_ASSERT(enkf_node->copy);  
  {
    const enkf_node_type * src = enkf_node;
    enkf_node_type * target;
    target = enkf_node_alloc(src->config);
    src->copy( src->data , target->data );  /* Calling the low level copy function */
    return target;
  }
}



bool enkf_node_include_type(const enkf_node_type * enkf_node, int mask) {
  return enkf_config_node_include_type(enkf_node->config , mask);
}


enkf_impl_type enkf_node_get_impl_type(const enkf_node_type * enkf_node) {
  return enkf_config_node_get_impl_type(enkf_node->config);
}


enkf_var_type enkf_node_get_var_type(const enkf_node_type * enkf_node) {
  return enkf_config_node_get_var_type(enkf_node->config);
}





void * enkf_node_value_ptr(const enkf_node_type * enkf_node) {
  return enkf_node->data;
}



/**
   This function calls the node spesific ecl_write function. IF the
   ecl_file of the (node == NULL) *ONLY* the path is sent to the node
   spesific file.
*/

void enkf_node_ecl_write(const enkf_node_type *enkf_node , const char *path , fortio_type * restart_fortio , int report_step) {
  if (enkf_node->ecl_write != NULL) {
    char * node_eclfile = enkf_config_node_alloc_outfile(enkf_node->config , report_step); /* Will return NULL if the node does not have any outfile format. */
    /*
      If the node does not have a outfile (i.e. ecl_file), the
      ecl_write function will be called with file argument NULL. It
      is then the responsability of the low-level implementation to
      do "the right thing".
    */
    enkf_node->ecl_write(enkf_node->data , path , node_eclfile , restart_fortio);
    util_safe_free( node_eclfile );
  }
}



/**
   This function takes a string - key - as input an calls a node
   specific function to look up one scalar based on that key. The key
   is always a string, but the the type of content will vary for the
   different objects. For a field, the key will be a string of "i,j,k"
   for a cell, for a multflt object the key will be the name of a
   fault, and so on.

   If the user has asked for something which does not exist the
   function SHOULD NOT FAIL. It should set *valid = false, and return
   0. To check the value of *valid is the responsability of the
   calling scope.
*/


double enkf_node_user_get(enkf_node_type * enkf_node , const char * key , bool * valid) {
  if (enkf_node->user_get != NULL) 
    return enkf_node->user_get(enkf_node->data , key , valid);
  else {
    fprintf(stderr , "** Warning: node:%s has no user_get implementation \n", enkf_node->node_key);
    *valid = false;
    return 0.0;
  }
}


void enkf_node_fload( enkf_node_type * enkf_node , const char * filename ) {
  FUNC_ASSERT( enkf_node->fload );
  enkf_node->fload( enkf_node->data , filename );
}





/**
   This function loads (internalizes) ECLIPSE results, the ecl_file
   instance with restart data, and the ecl_sum instance with summary
   data must already be loaded by the calling function.

   IFF the enkf_node has registered a filename to load from, that is
   passed to the specific load function, otherwise the run_path is sent
   to the load function.

   If the node does not have a ecl_load function, the function just
   returns.
*/


bool enkf_node_ecl_load(enkf_node_type *enkf_node , const char * run_path , const ecl_sum_type * ecl_sum, const ecl_file_type * restart_block , int report_step, int iens ) {
  bool loadOK;
  FUNC_ASSERT(enkf_node->ecl_load);
  {
    if (enkf_node_get_impl_type(enkf_node) == SUMMARY)
      /* Fast path for loading summary data. */
      loadOK = enkf_node->ecl_load(enkf_node->data , NULL  , ecl_sum , restart_block , report_step);
    else {
      char * input_file = enkf_config_node_alloc_infile(enkf_node->config , report_step);
      
      if (input_file != NULL) {
        char * file = util_alloc_filename( run_path , input_file , NULL);
        loadOK = enkf_node->ecl_load(enkf_node->data , file  , ecl_sum , restart_block , report_step);
        free(file);
      } else
        loadOK = enkf_node->ecl_load(enkf_node->data , run_path , ecl_sum , restart_block , report_step);
      
      util_safe_free( input_file );
    }
  }
  enkf_node->__report_step = report_step;
  enkf_node->__state       = FORECAST;
  enkf_node->__modified    = false;
  enkf_node->__iens        = iens; 
  return loadOK;
}



void enkf_node_ecl_load_static(enkf_node_type * enkf_node , const ecl_kw_type * ecl_kw, int report_step, int iens) {
  ecl_static_kw_init(enkf_node_value_ptr(enkf_node) , ecl_kw);
  enkf_node->__report_step 	= report_step;
  enkf_node->__state       	= FORECAST;
  enkf_node->__modified    	= false;
  enkf_node->__iens             = iens;
}


/**
   This function compares the internal __report_step with the input
   report_step, and return true if they are equal. It is used in the
   calling scope to discard static nodes which are no longer in use.
*/



bool enkf_node_store(enkf_node_type *enkf_node , buffer_type * buffer , bool internal_state , int report_step , int iens , state_enum state) {
  bool data_written = false;
  FUNC_ASSERT(enkf_node->store);
  data_written = enkf_node->store(enkf_node->data , buffer , report_step , internal_state);
  
  enkf_node->__report_step = report_step;
  enkf_node->__state       = state;
  enkf_node->__modified    = false;
  enkf_node->__iens        = iens;
  return data_written;
}



///**
//   Returns true if data is written to disk. If no data is written to
//   disk, and the function returns false, the calling scope is free
//   to skip storage (i.e. unlink an empty file).
//*/
//
//bool enkf_node_fwrite(enkf_node_type *enkf_node , FILE *stream , bool internal_state , int report_step , int iens , state_enum state) {
//  if (!enkf_node->__memory_allocated)
//    util_abort("%s: fatal internal error: tried to save node:%s - memory is not allocated - aborting.\n",__func__ , enkf_node->node_key);
//  {
//    bool data_written = false;
//    FUNC_ASSERT(enkf_node->fwrite_f);
//    data_written = enkf_node->fwrite_f(enkf_node->data , stream , internal_state);
//    
//    enkf_node->__report_step = report_step;
//    enkf_node->__state       = state;
//    enkf_node->__modified    = false;
//    enkf_node->__iens        = iens;
//    return data_written;
//  }
//}



void enkf_node_set_data(enkf_node_type *enkf_node , const void * data , int report_step , int iens , state_enum state) {
  FUNC_ASSERT(enkf_node->set_data);
  enkf_node->set_data(enkf_node->data , data);
    
  enkf_node->__report_step = report_step;
  enkf_node->__state       = state;
  enkf_node->__modified    = false;
  enkf_node->__iens        = iens;
}




void enkf_node_load(enkf_node_type *enkf_node , buffer_type * buffer , int report_step , int iens , state_enum state) {
  if ((report_step == enkf_node->__report_step) && (state == enkf_node->__state) && (enkf_node->__iens == iens) && (!enkf_node->__modified)) 
    return;  /* The in memory representation agrees with the buffer values */
  
  {
    FUNC_ASSERT(enkf_node->load);
    enkf_node->load(enkf_node->data , buffer , report_step);
    enkf_node->__modified    = false;
    enkf_node->__report_step = report_step;
    enkf_node->__state       = state;
    enkf_node->__iens        = iens;
  }
}




//void enkf_node_fread(enkf_node_type *enkf_node , FILE * stream , int report_step , int iens , state_enum state) {
//  if ((report_step == enkf_node->__report_step) && (state == enkf_node->__state) && (enkf_node->__iens == iens) && (!enkf_node->__modified))
//    return;  /* The in memory representation agrees with the disk image */
//
//  {
//    FUNC_ASSERT(enkf_node->fread_f);
//    enkf_node_ensure_memory(enkf_node);
//    enkf_node->fread_f(enkf_node->data , stream);
//    enkf_node->__modified    = false;
//    enkf_node->__report_step = report_step;
//    enkf_node->__state       = state;
//    enkf_node->__iens        = iens;
//  }
//}



void enkf_node_matrix_serialize(enkf_node_type *enkf_node , const active_list_type * active_list , matrix_type * A , int row_offset , int column) {
  FUNC_ASSERT(enkf_node->matrix_serialize);
  enkf_node->matrix_serialize(enkf_node->data , active_list , A , row_offset , column);
}



void enkf_node_matrix_deserialize(enkf_node_type *enkf_node , const active_list_type * active_list , const matrix_type * A , int row_offset , int column) {
  FUNC_ASSERT(enkf_node->matrix_deserialize);
  enkf_node->matrix_deserialize(enkf_node->data , active_list , A , row_offset , column);
  enkf_node->__modified = true;
}



void enkf_node_set_inflation( enkf_node_type * inflation , const enkf_node_type * std , const enkf_node_type * min_std) {
  {
    enkf_node_type * enkf_node = inflation;
    FUNC_ASSERT(enkf_node->set_inflation);
  }
  inflation->set_inflation( inflation->data , std->data , min_std->data );
}


void enkf_node_sqrt(enkf_node_type *enkf_node) {
  FUNC_ASSERT(enkf_node->isqrt);
  enkf_node->isqrt(enkf_node->data);
  enkf_node->__modified = true;
}


void enkf_node_scale(enkf_node_type *enkf_node , double scale_factor) {
  FUNC_ASSERT(enkf_node->scale);
  enkf_node->scale(enkf_node->data , scale_factor);
  enkf_node->__modified = true;
}


void enkf_node_iadd(enkf_node_type *enkf_node , const enkf_node_type * delta_node) {
  FUNC_ASSERT(enkf_node->iadd);
  enkf_node->iadd(enkf_node->data , delta_node->data);
  enkf_node->__modified = true;
}


void enkf_node_iaddsqr(enkf_node_type *enkf_node , const enkf_node_type * delta_node) {
  FUNC_ASSERT(enkf_node->iaddsqr);
  enkf_node->iaddsqr(enkf_node->data , delta_node->data);
  enkf_node->__modified = true;
}


void enkf_node_imul(enkf_node_type *enkf_node , const enkf_node_type * delta_node) {
  FUNC_ASSERT(enkf_node->imul);
  enkf_node->imul(enkf_node->data , delta_node->data);
  enkf_node->__modified = true;
}



/**
   The return value is whether any initialization has actually taken
   place. If the function returns false it is for instance not
   necessary to internalize anything.
*/

bool enkf_node_initialize(enkf_node_type *enkf_node, int iens) {
  if (enkf_node->initialize != NULL) {
    if (enkf_node->initialize(enkf_node->data , iens)) {
      enkf_node->__report_step = 0;
      enkf_node->__state       = ANALYZED;
      enkf_node->__modified    = true;
      return true;
    } else 
      return false; /* No init performed */
  } else
    return false;  /* No init performed */
}


/**
   Only the STATIC keywords actually support this operation.
*/

void enkf_node_free_data(enkf_node_type * enkf_node) {
  FUNC_ASSERT(enkf_node->free_data);
  enkf_node->free_data(enkf_node->data);
  enkf_node->__state            = UNDEFINED;
  enkf_node->__modified         = true;
  enkf_node->__report_step      = -1;  
}



void enkf_node_clear(enkf_node_type *enkf_node) {
  FUNC_ASSERT(enkf_node->clear);
  enkf_node->clear(enkf_node->data);
}


void enkf_node_free(enkf_node_type *enkf_node) {
  if (enkf_node->freef != NULL)
    enkf_node->freef(enkf_node->data);
  free(enkf_node->node_key);
  free(enkf_node);
}


void enkf_node_free__(void *void_node) {
  enkf_node_free((enkf_node_type *) void_node);
}

const char *enkf_node_get_key(const enkf_node_type * enkf_node) {
  return enkf_node->node_key;
}


#undef FUNC_ASSERT



/*****************************************************************/

/**
   This function has been implemented to ensure/force a reload of
   nodes when the case has changed.
*/
   
void enkf_node_invalidate_cache( enkf_node_type * node ) {
  node->__modified         = true;
  node->__report_step      = -1;
  node->__iens             = -1;
  node->__state            = UNDEFINED;
}


/* Manual inheritance - .... */
static enkf_node_type * enkf_node_alloc_empty(const enkf_config_node_type *config) {
  const char *node_key     = enkf_config_node_get_key(config);
  enkf_impl_type impl_type = enkf_config_node_get_impl_type(config);
  enkf_node_type * node    = util_malloc(sizeof * node , __func__);
  node->config             = config;
  node->node_key           = util_alloc_string_copy(node_key);
  node->data               = NULL;
  enkf_node_invalidate_cache( node );

  /*
    Start by initializing all function pointers to NULL.
  */
  //node->realloc_data   	   = NULL;
  node->alloc          	   = NULL;
  node->ecl_write      	   = NULL;
  node->ecl_load       	   = NULL;
  node->copy          	   = NULL;
  node->initialize     	   = NULL;
  node->freef          	   = NULL;
  node->free_data      	   = NULL;
  node->user_get       	   = NULL;
  node->set_data           = NULL;
  node->fload              = NULL; 
  node->load               = NULL;
  node->store              = NULL;
  node->matrix_serialize   = NULL; 
  node->matrix_deserialize = NULL;
  node->clear              = NULL;
  node->set_inflation      = NULL;

  /* Math operations: */
  node->iadd               = NULL;
  node->scale              = NULL;
  node->isqrt              = NULL;
  node->iaddsqr            = NULL;
  node->imul               = NULL;

  switch (impl_type) {
  case(GEN_KW):
    //node->realloc_data 	     = gen_kw_realloc_data__;
    node->alloc        	     = gen_kw_alloc__;
    node->ecl_write    	     = gen_kw_ecl_write__;
    node->copy        	     = gen_kw_copy__;
    node->initialize   	     = gen_kw_initialize__;
    node->freef        	     = gen_kw_free__;
    //node->free_data    	     = gen_kw_free_data__;
    node->user_get           = gen_kw_user_get__; 
    node->store              = gen_kw_store__;
    node->load               = gen_kw_load__;
    node->matrix_serialize   = gen_kw_matrix_serialize__;
    node->matrix_deserialize = gen_kw_matrix_deserialize__;
    node->clear              = gen_kw_clear__;
    node->iadd               = gen_kw_iadd__;
    node->scale              = gen_kw_scale__;
    node->iaddsqr            = gen_kw_iaddsqr__;
    node->imul               = gen_kw_imul__;
    node->isqrt              = gen_kw_isqrt__;
    node->set_inflation      = gen_kw_set_inflation__;
    node->fload              = gen_kw_fload__;
    break;
  case(SUMMARY):
    node->ecl_load           = summary_ecl_load__;
    //node->realloc_data       = summary_realloc_data__;
    node->alloc              = summary_alloc__;
    node->copy               = summary_copy__;
    node->freef              = summary_free__;
    //node->free_data          = summary_free_data__;
    node->user_get           = summary_user_get__; 
    node->load               = summary_load__;
    node->store              = summary_store__;
    node->matrix_serialize   = summary_matrix_serialize__;
    node->matrix_deserialize = summary_matrix_deserialize__;
    node->clear              = summary_clear__;
    node->iadd               = summary_iadd__;
    node->scale              = summary_scale__;
    node->iaddsqr            = summary_iaddsqr__;
    node->imul               = summary_imul__;
    node->isqrt              = summary_isqrt__;
    break;
  case(FIELD):
    //node->realloc_data 	     = field_realloc_data__;
    node->alloc        	     = field_alloc__;
    node->ecl_write    	     = field_ecl_write__; 
    node->ecl_load     	     = field_ecl_load__;  
    node->copy        	     = field_copy__;
    node->initialize   	     = field_initialize__;
    node->freef        	     = field_free__;
    //node->free_data    	     = field_free_data__;
    node->user_get     	     = field_user_get__;
    node->load         	     = field_load__;
    node->store        	     = field_store__;
    node->matrix_serialize   = field_matrix_serialize__;
    node->matrix_deserialize = field_matrix_deserialize__;

    node->clear              = field_clear__; 
    node->set_inflation      = field_set_inflation__;
    node->iadd               = field_iadd__;
    node->scale              = field_scale__;
    node->iaddsqr            = field_iaddsqr__;
    node->imul               = field_imul__; 
    node->isqrt              = field_isqrt__;
    node->fload              = field_fload__;
    break;
  case(STATIC):
    //node->realloc_data = ecl_static_kw_realloc_data__;
    node->ecl_write    = ecl_static_kw_ecl_write__; 
    node->alloc        = ecl_static_kw_alloc__;
    node->copy        = ecl_static_kw_copy__;
    node->freef        = ecl_static_kw_free__;
    node->free_data    = ecl_static_kw_free_data__;
    node->load         = ecl_static_kw_load__;
    node->store        = ecl_static_kw_store__;
    break;
  case(GEN_DATA):
    //node->realloc_data 	     = gen_data_realloc_data__;
    node->alloc        	     = gen_data_alloc__;
    node->initialize   	     = gen_data_initialize__;
    node->copy        	     = gen_data_copy__;
    node->freef        	     = gen_data_free__;
    //node->free_data    	     = gen_data_free_data__;
    node->ecl_write    	     = gen_data_ecl_write__;
    node->ecl_load     	     = gen_data_ecl_load__;
    node->user_get     	     = gen_data_user_get__;
    node->load         	     = gen_data_load__;
    node->store        	     = gen_data_store__;
    node->matrix_serialize   = gen_data_matrix_serialize__;
    node->matrix_deserialize = gen_data_matrix_deserialize__;
    node->set_inflation      = gen_data_set_inflation__;

    node->clear              = gen_data_clear__; 
    node->iadd               = gen_data_iadd__;
    node->scale              = gen_data_scale__;
    node->iaddsqr            = gen_data_iaddsqr__;
    node->imul               = gen_data_imul__;
    node->isqrt              = gen_data_isqrt__;
    node->fload              = gen_data_fload__;
    break;
  default:
    util_abort("%s: implementation type: %d unknown - all hell is loose - aborting \n",__func__ , impl_type);
  }
  return node;
}



#define CASE_SET(type , func) case(type): has_func = (func != NULL); break;
bool enkf_node_has_func(const enkf_node_type * node , node_function_type function_type) {
  bool has_func = false;
  switch (function_type) {
    CASE_SET(alloc_func        		    , node->alloc);
    CASE_SET(ecl_write_func    		    , node->ecl_write);
    CASE_SET(ecl_load_func                  , node->ecl_load);
    CASE_SET(copy_func        		    , node->copy);
    CASE_SET(initialize_func   		    , node->initialize);
    CASE_SET(free_func         		    , node->freef);
  default:
    fprintf(stderr,"%s: node_function_identifier: %d not recognized - aborting \n",__func__ , function_type);
  }
  return has_func;
}
#undef CASE_SET



enkf_node_type * enkf_node_alloc(const enkf_config_node_type * config) {
  enkf_node_type * node    = enkf_node_alloc_empty(config);
  enkf_node_alloc_domain_object(node);
  return node;
}


enkf_node_type * enkf_node_alloc_with_data(const enkf_config_node_type * config , void * data) {
  enkf_node_type * node    = enkf_node_alloc_empty( config );
  node->data               = data;
  return node;
}




bool enkf_node_internalize(const enkf_node_type * node, int report_step) {
  return enkf_config_node_internalize( node->config , report_step );
}


/*****************************************************************/

void enkf_node_upgrade_file_103( const char * path , const char * file , enkf_impl_type impl_type , int perc_complete , msg_type * msg) {
  if (strstr( file , "backup") != NULL)  /* This is a backup-file from a previous (halted) upgrade */
    return;
  
  {
    char * filename = util_alloc_filename( path , file , NULL);
    {
      char * msg_txt = util_alloc_sprintf("[%2d %s]  %s", perc_complete , "%" , filename);
      msg_update( msg , msg_txt );
      free(msg_txt);
    }
    {
      char * backup_path = util_alloc_filename(path , ".backup" , NULL);
      char * backup_file = util_alloc_filename(backup_path , file , NULL);
      
      if (!util_file_exists( backup_file )) {
	util_make_path( backup_path );
	util_copy_file( filename , backup_file );
	
	switch (impl_type) {
	case(GEN_KW):
	  gen_kw_upgrade_103( filename );
	  break;
	case(SUMMARY):
	  summary_upgrade_103( filename );
	  break;
	case(FIELD):
	  field_upgrade_103( filename );
	  break;
	case(STATIC):
	  ecl_static_kw_upgrade_103( filename );
	  break;
	case(GEN_DATA):
	  gen_data_upgrade_103( filename );
	  break;
	default:
	  break;
	}
      }
      
      free( backup_path);
      free( backup_file);
    }
    free(filename);
  }
}

/*****************************************************************/


ecl_write_ftype * enkf_node_get_func_pointer( const enkf_node_type * node ) {
  return node->ecl_write;
}
