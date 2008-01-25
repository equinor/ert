#include <string.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <enkf_node.h>
#include <enkf_config_node.h>
#include <util.h>
#include <multz.h>
#include <multflt.h>
#include <equil.h>
#include <field.h>
#include <well.h>
#include <ecl_static_kw.h>
#include <pgbox.h>

typedef struct serial_state_struct serial_state_type;
typedef enum   {forecast , serialized , analyzed} state_enum;


struct serial_state_struct {
  int        internal_offset;
  int        serial_size;
  size_t     offset;
  bool       state_complete;
  state_enum state;
};


struct enkf_node_struct {
  alloc_ftype         *alloc;
  fread_alloc_ftype   *fread_alloc;
  ecl_write_ftype     *ecl_write;
  fread_ftype         *fread_f;
  fwrite_ftype        *fwrite_f;
  swapin_ftype        *swapin;
  swapout_ftype       *swapout;

  serialize_ftype    *serialize;
  deserialize_ftype  *deserialize;
  
  sample_ftype       *sample;
  free_ftype         *freef;
  clear_ftype        *clear;
  copyc_ftype        *copyc;
  scale_ftype        *scale;
  iadd_ftype         *iadd;
  imul_ftype         *imul;
  isqrt_ftype        *isqrt;
  iaddsqr_ftype      *iaddsqr;

  serial_state_type  *serial_state;
  char               *node_key;
  void               *data;
  bool                swapped;
  const enkf_config_node_type *config;
};




const enkf_config_node_type * enkf_node_get_config(const enkf_node_type * node) { 
  return node->config; 
}

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
  /*
    printf("Oppdater forecast: offset:%d   elements:%d  complete:%d \n",offset,elements_added , complete);
  */
}



static void serial_state_update_serialized(serial_state_type * state , int new_internal_offset) {
  /*
    printf("Oppdaterer serialized: new_internal_offset:%d    complete:%d \n",new_internal_offset , state->state_complete);
  */
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

void enkf_node_realloc_data(enkf_node_type * node) {
  if (node->data != NULL)
    node->freef(node->data);

  node->data = node->alloc(enkf_config_node_get_ref(node->config));
}





/*
enkf_node_type * enkf_node_alloc_old(const char *node_key, 
				 const enkf_config_node_type * config    , 
				 alloc_ftype        * alloc     , 
				 ecl_write_ftype    * ecl_write , 
				 ens_read_ftype     * ens_read  , 
				 ens_write_ftype    * ens_write , 
				 swapout_ftype      * swapout   , 
				 swapin_ftype       * swapin    ,
				 copyc_ftype        * copyc     ,
				 sample_ftype       * sample    , 
				 serialize_ftype    * serialize , 
				 deserialize_ftype  * deserialize , 
				 free_ftype         * freef) {
  
  enkf_node_type *node = malloc(sizeof *node);
  node->alloc     = alloc;
  node->ecl_write = ecl_write;
  node->ens_read  = ens_read;
  node->ens_write = ens_write;
  node->swapin    = swapin;
  node->swapout   = swapout;
  node->sample    = sample;
  node->freef     = freef;
  node->copyc     = copyc;
  node->config    = config;
  node->node_key  = util_alloc_string_copy(node_key);
  node->data      = NULL;

  node->serialize    = serialize;
  node->deserialize  = deserialize;
  node->serial_state = serial_state_alloc();
  enkf_node_realloc_data(node);
  return node;
}
*/



void enkf_node_clear_serial_state(enkf_node_type * node) {
  serial_state_clear(node->serial_state);
}



enkf_node_type * enkf_node_copyc(const enkf_node_type * src) {
  if (src->copyc == NULL) {
    printf("Har ikke copyc funksjon\n");
    abort();
  }
  {
    enkf_node_type * new = enkf_node_alloc(enkf_node_get_key_ref(src) , src->config);
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


bool enkf_node_swapped(const enkf_node_type *enkf_node) {
  return enkf_node->swapped;
}



#define FUNC_ASSERT(func,func_name) if (func == NULL) { fprintf(stderr,"%s: function handler: %s not registered when writing node:%s - aborting\n",__func__ , func_name , enkf_node->node_key); abort(); }

void * enkf_node_value_ptr(const enkf_node_type * enkf_node) { 
  return enkf_node->data; 
}


void enkf_node_ecl_write(const enkf_node_type *enkf_node , const char *path) {
  FUNC_ASSERT(enkf_node->ecl_write , "ecl_write");
  enkf_node->ecl_write(enkf_node->data , path);
}

void enkf_node_fwrite(const enkf_node_type *enkf_node , FILE *stream) {
  FUNC_ASSERT(enkf_node->fwrite_f , "fwrite_f");
  enkf_node->fwrite_f(enkf_node->data , stream);
}


void enkf_node_fread(enkf_node_type *enkf_node , FILE * stream) {
  FUNC_ASSERT(enkf_node->fread_f , "fread_f");
  enkf_node->fread_f(enkf_node->data , stream);
}



void enkf_node_ens_clear(enkf_node_type *enkf_node) {
  FUNC_ASSERT(enkf_node->clear , "clear");
  enkf_node->clear(enkf_node->data);
}


int enkf_node_serialize(enkf_node_type *enkf_node , size_t serial_data_size , double *serial_data , size_t stride , size_t offset , bool *complete) {
  FUNC_ASSERT(enkf_node->serialize , "serialize");
  if (serial_state_do_serialize(enkf_node->serial_state)) {
    int internal_offset = serial_state_get_internal_offset(enkf_node->serial_state);
    int elements_added  = enkf_node->serialize(enkf_node->data , internal_offset , serial_data_size , serial_data , stride , offset , complete);
 
    serial_state_update_forecast(enkf_node->serial_state , offset , elements_added , *complete);
    return elements_added;
  } return 0;
}



void enkf_node_deserialize(enkf_node_type *enkf_node , double *serial_data , size_t stride) {
  FUNC_ASSERT(enkf_node->serialize , "serialize");
  if (serial_state_do_deserialize(enkf_node->serial_state)) {
    int serial_size , internal_offset , new_internal_offset;
    size_t serial_offset;

    serial_state_init_deserialize(enkf_node->serial_state , &internal_offset , &serial_offset, &serial_size );
    new_internal_offset = enkf_node->deserialize(enkf_node->data , internal_offset , serial_size , serial_data , stride , serial_offset);
    serial_state_update_serialized(enkf_node->serial_state , new_internal_offset);
    
  }
}


void enkf_node_sqrt(enkf_node_type *enkf_node) {
  FUNC_ASSERT(enkf_node->isqrt , "sqrt");
  enkf_node->isqrt(enkf_node->data);
}

void enkf_node_scale(enkf_node_type *enkf_node , double scale_factor) {
  FUNC_ASSERT(enkf_node->scale , "scale");
  enkf_node->scale(enkf_node->data , scale_factor);
}

void enkf_node_iadd(enkf_node_type *enkf_node , const enkf_node_type * delta_node) {
  FUNC_ASSERT(enkf_node->iadd , "iadd");
  enkf_node->iadd(enkf_node->data , delta_node->data);
}

void enkf_node_iaddsqr(enkf_node_type *enkf_node , const enkf_node_type * delta_node) {
  FUNC_ASSERT(enkf_node->iaddsqr , "iaddsqr");
  enkf_node->iaddsqr(enkf_node->data , delta_node->data);
}

void enkf_node_imul(enkf_node_type *enkf_node , const enkf_node_type * delta_node) {
  FUNC_ASSERT(enkf_node->imul , "imul");
  enkf_node->imul(enkf_node->data , delta_node->data);
}


void enkf_node_sample(enkf_node_type *enkf_node) {
  FUNC_ASSERT(enkf_node->sample , "sample");
  enkf_node->sample(enkf_node->data);
}


void enkf_node_swapin(enkf_node_type *enkf_node , const char * filename) {
  FUNC_ASSERT(enkf_node->swapin , "swapin");
  if (enkf_node_swapped(enkf_node)) {
    FILE * stream  = util_fopen(filename , "r");
    enkf_node->swapin(enkf_node->data , stream);
    fclose(stream);
  }
}


void enkf_node_swapout(enkf_node_type *enkf_node , const char * filename) {
  FUNC_ASSERT(enkf_node->swapin , "swapout");
  FILE * stream  = util_fopen(filename , "r");
  enkf_node->swapout(enkf_node->data , stream);
  enkf_node->swapped = true;
  fclose(stream);
}




void enkf_node_clear(enkf_node_type *enkf_node) {
  FUNC_ASSERT(enkf_node->clear , "clear");
  enkf_node->clear(enkf_node->data);
}


void enkf_node_printf(const enkf_node_type *enkf_node) {
  printf("%s \n",enkf_node->node_key);
}

/*
  char * enkf_node_alloc_ensfile(const enkf_node_type *enkf_node , const char * path) {
  FUNC_ASSERT(enkf_node->alloc_ensfile , "alloc_ensfile");
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

const char *enkf_node_get_key_ref(const enkf_node_type * enkf_node) { return enkf_node->node_key; }
#undef FUNC_ASSERT



/*****************************************************************/




/* Manual inheritance - .... */
static enkf_node_type * enkf_node_alloc_empty(const char *node_key,  const enkf_config_node_type * config) {
  enkf_node_type * node = util_malloc(sizeof * node , __func__);
  node->config          = config;
  node->node_key        = util_alloc_string_copy(node_key);
  node->data            = NULL;
  node->swapped         = false;

  enkf_impl_type impl_type = enkf_config_node_get_impl_type(config);
  switch (impl_type) {
  case(MULTZ):
    node->alloc       = multz_alloc__;
    node->fread_alloc = NULL; /*multz_fread_alloc__;*/
    node->ecl_write   = multz_ecl_write__;
    node->fread_f     = multz_fread__;
    node->fwrite_f    = multz_fwrite__;
    node->swapout     = multz_swapout__;
    node->swapin      = multz_swapin__;
    node->copyc       = multz_copyc__;
    node->sample      = multz_sample__;
    node->serialize   = multz_serialize__;
    node->deserialize = multz_deserialize__;
    node->freef       = multz_free__;
    break;
  case(MULTFLT):
    node->alloc       = multflt_alloc__;
    node->fread_alloc = NULL; /*multflt_fread_alloc__;*/
    node->ecl_write   = multflt_ecl_write__;
    node->fread_f     = multflt_fread__;
    node->fwrite_f    = multflt_fwrite__;
    node->swapout     = multflt_swapout__;
    node->swapin      = multflt_swapin__;
    node->copyc       = multflt_copyc__;
    node->sample      = multflt_sample__;
    node->serialize   = multflt_serialize__;
    node->deserialize = multflt_deserialize__;
    node->freef       = multflt_free__;
    break;
  case(WELL):
    node->alloc       = well_alloc__;
    node->fread_alloc = NULL; /*well_fread_alloc__;*/
    node->ecl_write   = NULL;
    node->fread_f     = well_fread__;
    node->fwrite_f    = well_fwrite__;
    node->swapout     = well_swapout__;
    node->swapin      = well_swapin__;
    node->copyc       = well_copyc__;
    node->sample      = NULL;
    node->serialize   = well_serialize__;
    node->deserialize = well_deserialize__;
    node->freef       = well_free__;
    break;
  case(FIELD):
    node->alloc       = field_alloc__;
    node->fread_alloc = NULL; /*field_fread_alloc__;*/
    node->ecl_write   = field_ecl_write__;
    node->fread_f     = field_fread__;
    node->fwrite_f    = field_fwrite__;
    node->swapout     = field_swapout__;
    node->swapin      = field_swapin__;
    node->copyc       = field_copyc__;
    node->sample      = field_sample__;
    node->serialize   = field_serialize__;
    node->deserialize = field_deserialize__;
    node->freef       = field_free__;
    break;
  case(PGBOX):
    node->alloc       = pgbox_alloc__;
    node->fread_alloc = NULL; /*pgbox_fread_alloc__; */
    node->ecl_write   = NULL; /* pgbox_ecl_write__;  */
    node->fread_f     = pgbox_fread__;
    node->fwrite_f    = pgbox_fwrite__;
    node->swapout     = pgbox_swapout__;
    node->swapin      = pgbox_swapin__;
    node->copyc       = pgbox_copyc__;
    node->sample      = pgbox_sample__;
    node->serialize   = pgbox_serialize__;
    node->deserialize = pgbox_deserialize__;
    node->freef       = pgbox_free__;
    break;
  case(EQUIL):
    node->alloc       = equil_alloc__;
    node->fread_alloc = NULL; /*equil_fread_alloc__;*/
    node->ecl_write   = equil_ecl_write__;
    node->fread_f     = equil_fread__;
    node->fwrite_f    = equil_fwrite__;
    node->swapout     = equil_swapout__;
    node->swapin      = equil_swapin__;
    node->copyc       = equil_copyc__;
    node->sample      = equil_sample__;
    node->serialize   = equil_serialize__;
    node->deserialize = equil_deserialize__;
    node->freef       = equil_free__;
    break;
  case(STATIC):
    node->alloc       = ecl_static_kw_alloc__;
    node->fread_alloc = NULL; /* ecl_static_kw_fread_alloc__;*/
    node->ecl_write   = NULL; /* ecl_static_kw_ecl_write__; */
    node->fread_f     = ecl_static_kw_fread__;
    node->fwrite_f    = ecl_static_kw_fwrite__;
    node->swapout     = ecl_static_kw_swapout__;
    node->swapin      = ecl_static_kw_swapin__;
    node->copyc       = ecl_static_kw_copyc__;
    node->sample      = NULL; 
    node->serialize   = NULL; 
    node->deserialize = NULL;
    node->freef       = ecl_static_kw_free__;
    break;
    
  default:
    fprintf(stderr,"%s: implementation type: %d unknown - all hell is loose - aborting \n",__func__ , impl_type);
    abort();
  }
  node->serial_state = serial_state_alloc();
  return node;
}



enkf_node_type * enkf_node_alloc(const char *node_key,  const enkf_config_node_type * config) {
  enkf_node_type * node = enkf_node_alloc_empty(node_key , config);
  enkf_node_realloc_data(node);
  return node;
}



