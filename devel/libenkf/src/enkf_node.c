#include <string.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <enkf_node.h>
#include <enkf_config_node.h>
#include <util.h>
#include <multz.h>
#include <relperm.h>
#include <multflt.h>
#include <equil.h>
#include <field.h>
#include <well.h>
#include <summary.h>
#include <ecl_static_kw.h>
#include <gen_kw.h>
#include <path_fmt.h>
#include <havana_fault.h>
#include <gen_data.h>

/**
   A small illustration (says more than thousand words ...) of how the
   enkf_node, enkf_config_node, field[1] and field_config[1] objects are
   linked.


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
      \__|        |					      |
         \________/					      |
             |						      |
             |						      |
             |						      |
            \|/						      |
     							      |
     ================                       		     \|/
     |              |   o-----------
     |  ================           |                =====================
     |  |              |   o--------                |                   |
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


   [1]: field is just an example, and could be replaces with any of
        the enkf object types.
*/






typedef struct serial_state_struct serial_state_type;


struct serial_state_struct {
  int        internal_offset;
  int        serial_size;
  size_t     offset;
  bool       state_complete;
  state_enum state;
};



struct enkf_node_struct {
  alloc_ftype         *alloc;
  ecl_write_ftype     *ecl_write;
  ecl_load_ftype      *ecl_load;
  fread_ftype         *fread_f;
  fwrite_ftype        *fwrite_f;
  realloc_data_ftype  *realloc_data;
  free_data_ftype     *free_data;

  serialize_ftype    *serialize;
  deserialize_ftype  *deserialize;

  initialize_ftype   		 * initialize;
  free_ftype         		 * freef;
  clear_ftype        		 * clear;
  copyc_ftype        		 * copyc;
  scale_ftype        		 * scale;
  iadd_ftype         		 * iadd;
  imul_ftype         		 * imul;
  isqrt_ftype        		 * isqrt;
  iaddsqr_ftype      		 * iaddsqr;
  ensemble_fprintf_results_ftype * fprintf_results;
  iget_ftype                     * iget;               /* The two last are a pair - the very last is interactive and used first. */
  get_index_ftype                * get_index;
  
  
  serial_state_type  *serial_state;
  char               *node_key;  /* Bør vel bare være i config node */
  void               *data;
  bool                memory_allocated;
  bool                modified;
  const enkf_config_node_type *config;
};




const enkf_config_node_type * enkf_node_get_config(const enkf_node_type * node) {
  return node->config;
}

/**
This function returns a pointer to the ensfile of the node. Observe
that nodes representing static ECLIPSE keywords do not have config
information, so in this case the hash_key is returned. It is important
that one does *NOT* return the ecl_kw_header_ref() of the static
keyword, because that might contain characters (typically '/') which
can not be part of a filename.
*/

/*
const char * enkf_node_get_ensfile_ref(const enkf_node_type * node ) {
  if (node->config == NULL)
    return node->node_key;
  else
    return enkf_config_node_get_ensfile_ref(node->config);
}
*/


const char     *  enkf_node_get_eclfile_ref(const enkf_node_type * node ) { return enkf_config_node_get_eclfile_ref(node->config); }



/*****************************************************************/

/*
  1. serialize: input : internal_offset
                output: elements_added , state_complete

  2. serial_state_update_forecast()

  3. EnkF update multiply X * serial_state.

  4. deserialize: input:  elements_added
                  output: updated internal_offste , state_complete

  5. serial_state_update_serialized()
*/





static void serial_state_clear(serial_state_type * state) {
  state->internal_offset    = 0;
  state->state              = forecast;
  state->serial_size        = 0;
  state->state_complete     = false;
  state->offset             = 0;
}


static serial_state_type * serial_state_alloc() {
  serial_state_type * state = malloc(sizeof * state);
  serial_state_clear(state);
  return state;
}


static void serial_state_free(serial_state_type * state) {
  free(state);
  state = NULL;
}



static bool serial_state_do_serialize(const serial_state_type * state) {
  if (state->state == forecast)
    return true;
  else
    return false;
}


static bool serial_state_do_deserialize(const serial_state_type * state) {
  if (state->state == serialized)
    return true;
  else
    return false;
}


static int serial_state_get_internal_offset(const serial_state_type * state) {
  return state->internal_offset;
}



static void serial_state_update_forecast(serial_state_type * state , size_t offset , int elements_added , bool complete) {
  state->serial_size    = elements_added;
  state->state_complete = complete;
  state->state          = serialized;
  state->offset         = offset;
}



static void serial_state_update_serialized(serial_state_type * state , int new_internal_offset) {
  if (state->state_complete) {
    state->state           = analyzed;
    state->serial_size     = -1;
    state->internal_offset = -1;
  } else {
    state->state           = forecast;
    state->serial_size     = -1;
    state->state_complete  = false;
    state->internal_offset = new_internal_offset;
  }
}


static void serial_state_init_deserialize(const serial_state_type * serial_state , int * internal_offset , size_t * serial_offset, int * serial_size) {
  *internal_offset = serial_state->internal_offset;
  *serial_offset   = serial_state->offset;
  *serial_size     = serial_state->serial_size;
}




/*****************************************************************/

/*
  All the function pointers REALLY should be in the config object ...
*/


void enkf_node_alloc_domain_object(enkf_node_type * node) {
  if (node->data != NULL)
    node->freef(node->data);
  node->data = node->alloc(enkf_config_node_get_ref(node->config));
  node->memory_allocated = true;
}




void enkf_node_clear_serial_state(enkf_node_type * node) {
  serial_state_clear(node->serial_state);
}



enkf_node_type * enkf_node_copyc(const enkf_node_type * src) {
  if (src->copyc == NULL) {
    printf("Har ikke copyc funksjon\n");
    abort();
  }
  {
    enkf_node_type * new;
    new = enkf_node_alloc(src->config);

    printf("%s: not properly implemented ... \n",__func__);
    abort();
    return new;
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



#define FUNC_ASSERT(func) \
   if (func == NULL) {      \
      fprintf(stderr,"%s: function handler: %s not registered for node:%s - aborting\n",__func__ , #func , enkf_node->node_key); \
      abort(); \
   }



void * enkf_node_value_ptr(const enkf_node_type * enkf_node) {
  return enkf_node->data;
}



/**
   This function calls the node spesific ecl_write function. IF the
   ecl_file of the (node == NULL) *ONLY* the path is sent to the node
   spesific file.

   This means that it is the responsibility of the node code to know
   wether a full file name is required, or only a path.
*/

void enkf_node_ecl_write(const enkf_node_type *enkf_node , const char *path) {
  FUNC_ASSERT(enkf_node->ecl_write);
  {
    const char * node_eclfile = enkf_config_node_get_eclfile_ref(enkf_node->config);
    if (node_eclfile != NULL) {
      char * target_file = util_alloc_full_path(path , node_eclfile);
      enkf_node->ecl_write(enkf_node->data , target_file);
      free(target_file);
    } else
      enkf_node->ecl_write(enkf_node->data , path);
  }
}


/**
   This function writes the node data, which must be either a field,
   or STATIC. Directly to an open fortio instance. No node->function is invoked.
*/

void enkf_node_ecl_write_fortio(const enkf_node_type *enkf_node , fortio_type * fortio , bool fmt_file , enkf_impl_type impl_type) {
  if (impl_type == STATIC) {
    ecl_kw_type * ecl_kw = ecl_static_kw_ecl_kw_ptr( enkf_node_value_ptr(enkf_node) );
    ecl_kw_set_fmt_file(ecl_kw , fmt_file);
    ecl_kw_fwrite(ecl_kw , fortio);
  } else if (impl_type == FIELD) {
    field_ecl_write1D_fortio(enkf_node_value_ptr(enkf_node) , fortio , fmt_file , fortio_get_endian_flip(fortio));
  } else
    util_abort("%s: internal error - unrecognized type:%d \n",__func__ , impl_type);
}

/**
   This function loads (internalizes) ECLIPSE results, the ecl_block
   instance with restart data, and the ecl_sum instance with summary
   data must be loaded by the calling function.

   If the node does not have a ecl_load function, the function just
   returns.
*/


void enkf_node_ecl_load(enkf_node_type *enkf_node , const char * run_path , const char * ecl_base , const ecl_sum_type * ecl_sum, int report_step) {
  FUNC_ASSERT(enkf_node->ecl_load);
  enkf_node->ecl_load(enkf_node->data , run_path   , ecl_base , ecl_sum , report_step);
}


void enkf_node_fwrite(const enkf_node_type *enkf_node , FILE *stream) {
  if (!enkf_node->memory_allocated)
    util_abort("%s: fatal internal error: tried to save node:%s - memory is not allocated - aborting.\n",__func__ , enkf_node->node_key);
  {
    FUNC_ASSERT(enkf_node->fwrite_f);
    enkf_node->fwrite_f(enkf_node->data , stream);
  }
}


void enkf_node_ensure_memory(enkf_node_type * enkf_node) {
  FUNC_ASSERT(enkf_node->realloc_data);
  if (!enkf_node->memory_allocated) {
    enkf_node->realloc_data(enkf_node->data);
    enkf_node->memory_allocated = true;
  }
}


void enkf_node_ensemble_fprintf_results(const enkf_node_type ** ensemble , int ens_size , int report_step , const char * path) {
  /*
    FUNC_ASSERT(ensemble[0]->fprintf_results);
  */
  {
    void ** data_pointers = util_malloc(ens_size * sizeof * data_pointers , __func__);
    char * filename       = util_alloc_full_path(path , ensemble[0]->node_key);
    int iens;
    for (iens=0; iens < ens_size; iens++)
      data_pointers[iens] = ensemble[iens]->data;

    ensemble[0]->fprintf_results((const void **) data_pointers , ens_size , filename);
    free(filename);
    free(data_pointers);
  }
}


bool enkf_node_memory_allocated(const enkf_node_type * node) { return node->memory_allocated; }

static void enkf_node_assert_memory(const enkf_node_type * enkf_node , const char * caller) {
  if (!enkf_node_memory_allocated(enkf_node)) {
    printf("Fatal error - no memory ?? \n");
    util_abort("%s:  tried to call:%s without allocated memory for node:%s - internal ERROR - aborting.\n",__func__ , caller , enkf_node->node_key);
  }
}


void enkf_node_fread(enkf_node_type *enkf_node , FILE * stream) {
  FUNC_ASSERT(enkf_node->fread_f);
  enkf_node_ensure_memory(enkf_node);
  enkf_node->fread_f(enkf_node->data , stream);
}



void enkf_node_ens_clear(enkf_node_type *enkf_node) {
  FUNC_ASSERT(enkf_node->clear);
  enkf_node->clear(enkf_node->data);
}


int enkf_node_serialize(enkf_node_type *enkf_node , size_t serial_data_size , double *serial_data , size_t stride , size_t offset , bool *complete) {
  FUNC_ASSERT(enkf_node->serialize);
  enkf_node_assert_memory(enkf_node , __func__);
  if (serial_state_do_serialize(enkf_node->serial_state)) {
    int internal_offset = serial_state_get_internal_offset(enkf_node->serial_state);
    int elements_added  = enkf_node->serialize(enkf_node->data , internal_offset , serial_data_size , serial_data , stride , offset , complete);

    serial_state_update_forecast(enkf_node->serial_state , offset , elements_added , *complete);
    return elements_added;
  } return 0;
}



void enkf_node_deserialize(enkf_node_type *enkf_node , double *serial_data , size_t stride) {
  FUNC_ASSERT(enkf_node->serialize);
  if (serial_state_do_deserialize(enkf_node->serial_state)) {
    int serial_size , internal_offset , new_internal_offset;
    size_t serial_offset;

    serial_state_init_deserialize(enkf_node->serial_state , &internal_offset , &serial_offset, &serial_size );
    new_internal_offset = enkf_node->deserialize(enkf_node->data , internal_offset , serial_size , serial_data , stride , serial_offset);
    serial_state_update_serialized(enkf_node->serial_state , new_internal_offset);
  }
}


void enkf_node_sqrt(enkf_node_type *enkf_node) {
  FUNC_ASSERT(enkf_node->isqrt);
  enkf_node->isqrt(enkf_node->data);
}

void enkf_node_scale(enkf_node_type *enkf_node , double scale_factor) {
  FUNC_ASSERT(enkf_node->scale);
  enkf_node->scale(enkf_node->data , scale_factor);
}

void enkf_node_iadd(enkf_node_type *enkf_node , const enkf_node_type * delta_node) {
  FUNC_ASSERT(enkf_node->iadd);
  enkf_node->iadd(enkf_node->data , delta_node->data);
}

void enkf_node_iaddsqr(enkf_node_type *enkf_node , const enkf_node_type * delta_node) {
  FUNC_ASSERT(enkf_node->iaddsqr);
  enkf_node->iaddsqr(enkf_node->data , delta_node->data);
}

void enkf_node_imul(enkf_node_type *enkf_node , const enkf_node_type * delta_node) {
  FUNC_ASSERT(enkf_node->imul);
  enkf_node->imul(enkf_node->data , delta_node->data);
}


void enkf_node_initialize(enkf_node_type *enkf_node, int iens) {
  FUNC_ASSERT(enkf_node->initialize);
  enkf_node->initialize(enkf_node->data , iens);
}


void enkf_node_free_data(enkf_node_type * enkf_node) {
  FUNC_ASSERT(enkf_node->free_data);
  enkf_node->free_data(enkf_node->data);
  enkf_node->memory_allocated = false;
}



void enkf_node_clear(enkf_node_type *enkf_node) {
  FUNC_ASSERT(enkf_node->clear);
  enkf_node->clear(enkf_node->data);
}


void enkf_node_printf(const enkf_node_type *enkf_node) {
  printf("%s \n",enkf_node->node_key);
}

/*
  char * enkf_node_alloc_ensfile(const enkf_node_type *enkf_node , const char * path) {
  FUNC_ASSERT(enkf_node , "alloc_ensfile");
  return enkf_node->alloc_ensfile(enkf_node->data , path);
}
*/

void enkf_node_free(enkf_node_type *enkf_node) {
  if (enkf_node->freef != NULL)
    enkf_node->freef(enkf_node->data);
  free(enkf_node->node_key);
  serial_state_free(enkf_node->serial_state);
  free(enkf_node);
  enkf_node = NULL;
}


void enkf_node_free__(void *void_node) {
  enkf_node_free((enkf_node_type *) void_node);
}

const char *enkf_node_get_key_ref(const enkf_node_type * enkf_node) {
  return enkf_node->node_key;
}


#undef FUNC_ASSERT



/*****************************************************************/




/* Manual inheritance - .... */
static enkf_node_type * enkf_node_alloc_empty(const enkf_config_node_type *config) {
  const char *node_key     = enkf_config_node_get_key_ref(config);
  enkf_impl_type impl_type = enkf_config_node_get_impl_type(config);
  enkf_node_type * node  = util_malloc(sizeof * node , __func__);
  node->config           = config;
  node->node_key         = util_alloc_string_copy(node_key);
  node->data             = NULL;
  node->memory_allocated = false;

  /*
     Start by initializing all function pointers
     to NULL.
  */
  node->realloc_data   = NULL;
  node->alloc          = NULL;
  node->ecl_write      = NULL;
  node->ecl_load       = NULL;
  node->fread_f        = NULL;
  node->fwrite_f       = NULL;
  node->copyc          = NULL;
  node->initialize     = NULL;
  node->serialize      = NULL;
  node->deserialize    = NULL;
  node->freef          = NULL;
  node->free_data      = NULL;
  node->fprintf_results= NULL;
  node->iget           = NULL;
  node->get_index      = NULL;

  switch (impl_type) {
  case(GEN_KW):
    node->realloc_data 	  = gen_kw_realloc_data__;
    node->alloc        	  = gen_kw_alloc__;
    node->ecl_write    	  = gen_kw_ecl_write__;
    node->fread_f      	  = gen_kw_fread__;
    node->fwrite_f     	  = gen_kw_fwrite__;
    node->copyc        	  = gen_kw_copyc__;
    node->initialize   	  = gen_kw_initialize__;
    node->serialize    	  = gen_kw_serialize__;
    node->deserialize  	  = gen_kw_deserialize__;
    node->freef        	  = gen_kw_free__;
    node->free_data    	  = gen_kw_free_data__;
    node->fprintf_results = gen_kw_ensemble_fprintf_results__;
    break;
  case(MULTZ):
    node->realloc_data = multz_realloc_data__;
    node->alloc        = multz_alloc__;
    node->ecl_write    = multz_ecl_write__;
    node->fread_f      = multz_fread__;
    node->fwrite_f     = multz_fwrite__;
    node->copyc        = multz_copyc__;
    node->initialize   = multz_initialize__;
    node->serialize    = multz_serialize__;
    node->deserialize  = multz_deserialize__;
    node->freef        = multz_free__;
    node->free_data    = multz_free_data__;
    break;
  case(RELPERM):
    node->alloc       = relperm_alloc__;
    node->ecl_write   = relperm_ecl_write__;
    node->fread_f     = relperm_fread__;
    node->fwrite_f    = relperm_fwrite__;
    node->copyc       = relperm_copyc__;
    node->initialize  = relperm_initialize__;
    node->serialize   = relperm_serialize__;
    node->deserialize = relperm_deserialize__;
    node->freef       = relperm_free__;
    node->free_data   = relperm_free_data__;
    break;
  case(MULTFLT):
    node->realloc_data 	  = multflt_realloc_data__;
    node->alloc        	  = multflt_alloc__;
    node->ecl_write    	  = multflt_ecl_write__;
    node->fread_f      	  = multflt_fread__;
    node->fwrite_f     	  = multflt_fwrite__;
    node->copyc        	  = multflt_copyc__;
    node->initialize   	  = multflt_initialize__;
    node->serialize    	  = multflt_serialize__;
    node->deserialize  	  = multflt_deserialize__;
    node->freef        	  = multflt_free__;
    node->free_data    	  = multflt_free_data__;
    node->fprintf_results = multflt_ensemble_fprintf_results__;
    break;
  case(WELL):
    node->ecl_load        = well_ecl_load__;
    node->realloc_data 	  = well_realloc_data__;
    node->alloc        	  = well_alloc__;
    node->fread_f      	  = well_fread__;
    node->fwrite_f     	  = well_fwrite__;
    node->copyc        	  = well_copyc__;
    node->serialize    	  = well_serialize__;
    node->deserialize  	  = well_deserialize__;
    node->freef        	  = well_free__;
    node->free_data    	  = well_free_data__;
    node->fprintf_results = well_ensemble_fprintf_results__;
    break;
  case(SUMMARY):
    node->ecl_load     = summary_ecl_load__;
    node->realloc_data = summary_realloc_data__;
    node->alloc        = summary_alloc__;
    node->fread_f      = summary_fread__;
    node->fwrite_f     = summary_fwrite__;
    node->copyc        = summary_copyc__;
    node->serialize    = summary_serialize__;
    node->deserialize  = summary_deserialize__;
    node->freef        = summary_free__;
    node->free_data    = summary_free_data__;
    break;
  case(HAVANA_FAULT):
    node->realloc_data 	  = havana_fault_realloc_data__;
    node->alloc        	  = havana_fault_alloc__;
    node->ecl_write    	  = havana_fault_ecl_write__;
    node->fread_f      	  = havana_fault_fread__;
    node->fwrite_f     	  = havana_fault_fwrite__;
    node->copyc        	  = havana_fault_copyc__;
    node->serialize    	  = havana_fault_serialize__;
    node->deserialize  	  = havana_fault_deserialize__;
    node->freef        	  = havana_fault_free__;
    node->free_data    	  = havana_fault_free_data__;
    node->initialize   	  = havana_fault_initialize__;
    node->fprintf_results = havana_fault_ensemble_fprintf_results__;
    break;
  case(FIELD):
    node->realloc_data = field_realloc_data__;
    node->alloc        = field_alloc__;
    node->ecl_write    = field_ecl_write__; /* This is the function suitable for writing PORO / PERM +++ . Pressure ++ uses a field_xxx function directly. */
    node->fread_f      = field_fread__;
    node->fwrite_f     = field_fwrite__;
    node->copyc        = field_copyc__;
    node->initialize   = field_initialize__;
    node->serialize    = field_serialize__;
    node->deserialize  = field_deserialize__;
    node->freef        = field_free__;
    node->free_data    = field_free_data__;
    node->iget         = field_iget__;
    break;
  case(EQUIL):
    node->alloc       = equil_alloc__;
    node->ecl_write   = equil_ecl_write__;
    node->fread_f     = equil_fread__;
    node->fwrite_f    = equil_fwrite__;
    node->copyc       = equil_copyc__;
    node->initialize  = equil_initialize__;
    node->serialize   = equil_serialize__;
    node->deserialize = equil_deserialize__;
    node->freef       = equil_free__;
    node->free_data   = equil_free_data__;
    break;
  case(STATIC):
    node->realloc_data = ecl_static_kw_realloc_data__;
    node->ecl_write    = NULL; /* ecl_static_kw_ecl_write__; */
    node->alloc        = ecl_static_kw_alloc__;
    node->fread_f      = ecl_static_kw_fread__;
    node->fwrite_f     = ecl_static_kw_fwrite__;
    node->copyc        = ecl_static_kw_copyc__;
    node->freef        = ecl_static_kw_free__;
    node->free_data    = ecl_static_kw_free_data__;
    break;
  case(GEN_DATA):
    node->realloc_data = gen_data_realloc_data__;
    node->alloc        = gen_data_alloc__;
    node->fread_f      = gen_data_fread__;
    node->fwrite_f     = gen_data_fwrite__;
    node->copyc        = gen_data_copyc__;
    node->freef        = gen_data_free__;
    node->free_data    = gen_data_free_data__;
    node->ecl_load     = gen_data_ecl_load__;
    node->serialize    = gen_data_serialize__;
    node->deserialize  = gen_data_deserialize__;
    break;
  default:
    fprintf(stderr,"%s: implementation type: %d unknown - all hell is loose - aborting \n",__func__ , impl_type);
    abort();
  }
  node->serial_state = serial_state_alloc();
  return node;
}


void enkf_node_set_modified(enkf_node_type * node)      { node->modified = true; }
bool enkf_node_get_modified(const enkf_node_type *node) { return node->modified; }


#define CASE_SET(type , func) case(type): has_func = (func != NULL); break;
bool enkf_node_has_func(const enkf_node_type * node , node_function_type function_type) {
  bool has_func = false;
  switch (function_type) {
    CASE_SET(alloc_func        		    , node->alloc);
    CASE_SET(ecl_write_func    		    , node->ecl_write);
    CASE_SET(ecl_load_func                  , node->ecl_load);
    CASE_SET(fread_func        		    , node->fread_f);
    CASE_SET(fwrite_func       		    , node->fwrite_f);
    CASE_SET(copyc_func        		    , node->copyc);
    CASE_SET(initialize_func   		    , node->initialize);
    CASE_SET(serialize_func    		    , node->serialize);
    CASE_SET(deserialize_func  		    , node->deserialize);
    CASE_SET(free_func         		    , node->freef);
    CASE_SET(free_data_func    		    , node->free_data);
    CASE_SET(ensemble_fprintf_results_func  , node->fprintf_results);
  default:
    fprintf(stderr,"%s: node_function_identifier: %d not recognized - aborting \n",__func__ , function_type);
  }
  return has_func;
}
#undef CASE_SET


static enkf_node_type * enkf_node_alloc__(const enkf_config_node_type * config) {
  enkf_node_type * node    = enkf_node_alloc_empty(config);

  node = enkf_node_alloc_empty(config);
  enkf_node_alloc_domain_object(node);

  enkf_node_set_modified(node);
  return node;
}


enkf_node_type * enkf_node_alloc(const enkf_config_node_type * config) {
  if (config == NULL)
    util_abort("%s: internal error - must use enkf_node_alloc_static() to allocate static nodes.\n",__func__);

  return enkf_node_alloc__(config);
}







void enkf_node_load_static_ecl_kw(enkf_node_type * enkf_node , const ecl_kw_type * ecl_kw) {
  if (enkf_node_get_impl_type(enkf_node) != STATIC)
    util_abort("%s: internal error - this function should only be called with static nodes. \n" , __func__);

  ecl_static_kw_init(enkf_node_value_ptr(enkf_node) , ecl_kw);
  enkf_node->memory_allocated = true;
}

