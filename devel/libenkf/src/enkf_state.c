#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <list.h>
#include <hash.h>
#include <fortio.h>
#include <util.h>
#include <ecl_kw.h>
#include <ecl_block.h>
#include <ecl_fstate.h>
#include <list_node.h>
#include <enkf_node.h>
#include <enkf_state.h>
#include <enkf_config.h>
#include <enkf_types.h>
#include <ecl_static_kw.h>
#include <field.h>
#include <field_config.h>
#include <ecl_util.h>
#include <thread_pool.h>
#include <path_fmt.h>

#include <ecl_sum.h>
#include <well.h>
#include <multz.h>
#include <multflt.h>
#include <equil.h>
#include <well_config.h>
#include <void_arg.h>
#include <pgbox_config.h>
#include <pgbox.h>
#include <restart_kw_list.h>
#include <enkf_fs.h>
#include "enkf_config_decl.h"


struct enkf_state_struct {
  restart_kw_list_type  * restart_kw_list;
  list_type    	   	* node_list;
  hash_type    	   	* node_hash;
  hash_type    	   	* impl_types;
  
  enkf_fs_type          * enkf_fs;
  enkf_config_type 	* config;
  char             	* eclbase;
  char                  * run_path;
  bool             	 _fmt_file;  
  int                     my_iens;
};


static void enkf_state_add_node__2(enkf_state_type * , const char * , const enkf_node_type * );


/*****************************************************************/

#define ENKF_STATE_APPLY(node_func)                                      \
void enkf_state_ ## node_func(enkf_state_type * enkf_state , int mask) { \
  list_node_type *list_node;                                             \
  list_node = list_get_head(enkf_state->node_list);                      \
  while (list_node != NULL) {                                            \
    enkf_node_type *enkf_node = list_node_value_ptr(list_node);          \
    if (enkf_node_include_type(enkf_node , mask))                        \
      enkf_node_ ## node_func (enkf_node);                               \
    list_node = list_node_get_next(list_node);                           \
  }                                                                      \
}


void enkf_state_apply_NEW(enkf_state_type * enkf_state , enkf_node_ftype1 * node_func , int mask) {
  enkf_node_type * enkf_node;
  bool cont;
  enkf_node = hash_iter_get_first(enkf_state->node_hash , &cont);
  while (cont) {
    if (enkf_node_include_type(enkf_node , mask))                        
      node_func(enkf_node);                               
    enkf_node = hash_iter_get_next(enkf_state->node_hash , &cont);
  }                                                                      
}



void enkf_state_apply(enkf_state_type * enkf_state , enkf_node_ftype1 * node_func , int mask) {
  list_node_type *list_node;                                             
  list_node = list_get_head(enkf_state->node_list);                      
  while (list_node != NULL) {                                            
    enkf_node_type *enkf_node = list_node_value_ptr(list_node);          
    if (enkf_node_include_type(enkf_node , mask))                        
      node_func (enkf_node);                               
    list_node = list_node_get_next(list_node);                           
  }                                                                      
}


/*****************************************************************/

#define ENKF_STATE_APPLY2(node_func)                                     \
void enkf_state_ ## node_func(enkf_state_type * enkf_state , const enkf_state_type *enkf_state2 , int mask) { \
  list_node_type *list_node;                                            \
  list_node_type *list_node2;                                           \
  list_node  = list_get_head(enkf_state->node_list);                    \
  list_node2 = list_get_head(enkf_state2->node_list);                   \
  while (list_node != NULL) {                                           \
    enkf_node_type *enkf_node  = list_node_value_ptr(list_node);        \
    enkf_node_type *enkf_node2 = list_node_value_ptr(list_node2);       \
    if (enkf_node_include_type(enkf_node , mask))                       \
      enkf_node_ ## node_func (enkf_node , enkf_node2);                 \
    list_node  = list_node_get_next(list_node);                         \
    list_node2 = list_node_get_next(list_node2);                        \
  }                                                                     \
}

/*****************************************************************/

#define ENKF_STATE_APPLY_SCALAR(node_func)                                     \
void enkf_state_ ## node_func(enkf_state_type * enkf_state , double scalar, int mask) { \
  list_node_type *list_node;                                            \
  list_node  = list_get_head(enkf_state->node_list);                    \
  while (list_node != NULL) {                                           \
    enkf_node_type *enkf_node  = list_node_value_ptr(list_node);        \
    if (enkf_node_include_type(enkf_node , mask))                       \
      enkf_node_ ## node_func (enkf_node , scalar);                     \
    list_node  = list_node_get_next(list_node);                         \
  }                                                                     \
}

/*****************************************************************/

#define ENKF_STATE_APPLY_PATH(node_func)                                     \
void enkf_state_ ## node_func(enkf_state_type * enkf_state , const char *path, int mask) { \
  list_node_type *list_node;                                            \
  list_node  = list_get_head(enkf_state->node_list);                    \
  while (list_node != NULL) {                                           \
    enkf_node_type *enkf_node  = list_node_value_ptr(list_node);        \
    if (enkf_node_include_type(enkf_node , mask))                       \
      enkf_node_ ## node_func (enkf_node , path);                       \
    list_node  = list_node_get_next(list_node);                         \
  }                                                                     \
}

/*****************************************************************/

/*
  In principle different members can have fmt / binary files - overkill??
*/

bool enkf_state_fmt_file(const enkf_state_type * enkf_state) {
  return enkf_state->_fmt_file;
}

/*
  Because the ecl_sum / ecl_fstate routine takes an integer
  mode as input, instead of a bool.
*/

int enkf_state_fmt_mode(const enkf_state_type * enkf_state) {
  if (enkf_state_fmt_file(enkf_state))
    return ECL_FORMATTED;
  else
    return ECL_BINARY;
}


static const char * enkf_state_select_ens_path(const enkf_state_type * enkf_state , const char * node_kw , bool forecast) {
  fprintf(stderr,"%s not implemented - aborting \n",__func__);
  return NULL;
}


void enkf_state_set_run_path(enkf_state_type * enkf_state) {
  if (enkf_state->run_path != NULL)
    free(enkf_state->run_path);
  
  enkf_state->run_path        = path_fmt_alloc_path(enkf_state->config->run_path , enkf_state->my_iens);
}

void enkf_state_set_iens(enkf_state_type * enkf_state , int iens) {
  enkf_state->my_iens         = iens;
}



enkf_state_type *enkf_state_alloc(const enkf_config_type * config , const char * eclbase, int iens , enkf_fs_type * fs , bool fmt_file) {
  enkf_state_type * enkf_state = malloc(sizeof *enkf_state);
  
  enkf_state->config          = (enkf_config_type *) config;
  enkf_state->node_list       = list_alloc();
  enkf_state->node_hash       = hash_alloc(10);
  enkf_state->impl_types      = hash_alloc(10);
  enkf_state->eclbase         = util_alloc_string_copy(eclbase);
  enkf_state->_fmt_file       = fmt_file;
  enkf_state->restart_kw_list = restart_kw_list_alloc();
  enkf_state_set_iens(enkf_state , iens);
  enkf_state->run_path        = NULL;
  enkf_state_set_run_path(enkf_state);
  enkf_state->enkf_fs         = fs;
  /* 
     This information should really be in a config object. Currently
     not used, but could/will be used in a string based type lookup. 
  */
  hash_insert_int(enkf_state->impl_types     , "WELL"       , WELL);
  hash_insert_int(enkf_state->impl_types     , "MULTZ"      , MULTZ);
  hash_insert_int(enkf_state->impl_types     , "MULTFLT"    , MULTFLT);
  hash_insert_int(enkf_state->impl_types     , "EQUIL"      , EQUIL);
  hash_insert_int(enkf_state->impl_types     , "STATIC"     , STATIC);
  hash_insert_int(enkf_state->impl_types     , "FIELD"      , FIELD);
  
  return enkf_state;
}

/*
  hash_node -> list_node -> enkf_node -> {Actual enkf object: multz_type/equil_type/multflt_type/...}
*/




enkf_state_type * enkf_state_copyc(const enkf_state_type * src) {
  enkf_state_type * new = enkf_state_alloc(src->config , src->eclbase, src->my_iens , src->enkf_fs , enkf_state_fmt_file(src));
  list_node_type *list_node;                                          
  list_node = list_get_head(src->node_list);                     

  while (list_node != NULL) {                                           
    {
      enkf_node_type *enkf_node = list_node_value_ptr(list_node);         
      enkf_node_type *new_node  = enkf_node_copyc(enkf_node);
      enkf_state_add_node__2(new , enkf_node_get_key_ref(new_node) , new_node);
      list_node = list_node_get_next(list_node);                          
    }
  }
  
  return new;
}



static bool enkf_state_has_node(const enkf_state_type * enkf_state , const char * node_key) {
  return hash_has_key(enkf_state->node_hash , node_key);
}



static void enkf_state_add_node__2(enkf_state_type * enkf_state , const char * node_name , const enkf_node_type * node) {
  list_node_type *list_node = list_append_list_owned_ref(enkf_state->node_list , node , enkf_node_free__);
  /*
    The hash contains a pointer to a list_node structure, which contains a pointer
    to an enkf_node which contains a pointer to the actual enkf object.
  */
  hash_insert_ref(enkf_state->node_hash , node_name  , list_node);
}



static void enkf_state_add_node__1(enkf_state_type * enkf_state , const char * node_name , const enkf_config_node_type * config) {
  enkf_node_type *enkf_node = enkf_node_alloc(node_name , config);
  enkf_state_add_node__2(enkf_state , node_name , enkf_node);    

  /* All code below here is special code for plurigaussian fields */
  {
    enkf_impl_type impl_type = enkf_config_node_get_impl_type(config);
    if (impl_type == PGBOX) {
      const pgbox_config_type * pgbox_config = enkf_config_node_get_ref(config);
      const char * target_key = pgbox_config_get_target_key(pgbox_config);
      if (enkf_state_has_node(enkf_state , target_key)) {
	enkf_node_type * target_node = enkf_state_get_node(enkf_state , target_key);
	if (enkf_node_get_impl_type(target_node) != FIELD) {
	  fprintf(stderr,"%s: target node:%s is not of type field - aborting \n",__func__ , target_key);
	  abort();
	}
	pgbox_set_target_field(enkf_node_value_ptr(enkf_node) , enkf_node_value_ptr(target_node));
      } else {
	fprintf(stderr,"%s: target field:%s must be added to the state object *BEFORE* the pgbox object - aborting \n" , __func__ , target_key);
	abort();
      }
    }
  }
}



/* 
   Maybe this should just take implementation type
   as an input integer, instead of going via the string type_str ??
*/

void enkf_state_add_node(enkf_state_type * enkf_state , const char * node_name) {
  if (enkf_state_has_node(enkf_state , node_name)) {
    fprintf(stderr,"%s: node:%s already added  - aborting \n",__func__ , node_name);
    abort();
  }

  if (!enkf_config_has_key(enkf_state->config , node_name)) {
    fprintf(stderr,"%s could not find configuration object for:%s - aborting \n",__func__ , node_name);
    abort();
  }
  {
    const enkf_config_node_type *config  = enkf_config_get_ref(enkf_state->config  , node_name);
    enkf_state_add_node__1(enkf_state , node_name , config);
  }
}




static void enkf_state_load_ecl_restart__(enkf_state_type * enkf_state , const ecl_block_type *ecl_block) {
  ecl_kw_type * ecl_kw = ecl_block_get_first_kw(ecl_block);
  restart_kw_list_reset(enkf_state->restart_kw_list);
  while (ecl_kw != NULL) {
    char *kw                       = ecl_kw_alloc_strip_header(ecl_kw);
    const enkf_impl_type impl_type = enkf_config_impl_type(enkf_state->config , kw);
    enkf_var_type enkf_type;

    switch (impl_type) {
    case(FIELD):
      enkf_type = ecl_restart;
      break;
    case(STATIC):
      enkf_type = ecl_static;
      break;
    default:
      fprintf(stderr,"%s internal error - when loading ECLIPSE restart files only FIELD and STATIC implementation types are (currently) recognized - aborting \n",__func__);
      abort();
    }    
    restart_kw_list_add(enkf_state->restart_kw_list , kw);

    if (!enkf_config_has_key(enkf_state->config , kw)) 
      enkf_config_add_type0(enkf_state->config , kw , ecl_kw_get_size(ecl_kw) , enkf_type , impl_type);
    
    if (!enkf_state_has_node(enkf_state , kw)) 
      enkf_state_add_node__1(enkf_state , kw , enkf_config_get_ref(enkf_state->config , kw)); 
    
    {
      enkf_node_type * enkf_node = enkf_state_get_node(enkf_state , kw);
     
      switch (impl_type)  {
      case(FIELD):
	field_copy_ecl_kw_data(enkf_node_value_ptr(enkf_node) , ecl_kw);
	break;
      case(STATIC):
	ecl_static_kw_init(enkf_node_value_ptr(enkf_node) , ecl_kw);
	break;
      default:
	fprintf(stderr,"%s: internal error - can only get data from implementation types: FIELD and STATIC - aborting \n",__func__);
	abort();
      }

    }
    free(kw);
    ecl_kw = ecl_block_get_next_kw(ecl_block);
  }
}




void enkf_state_load_ecl_restart(enkf_state_type * enkf_state ,  bool unified , int report_step) {
  bool at_eof;
  const bool fmt_file  = enkf_state_fmt_file(enkf_state);
  bool endian_swap     = enkf_config_get_endian_swap(enkf_state->config);
  ecl_block_type       * ecl_block;
  char * restart_file  = ecl_util_alloc_exfilename(enkf_state->run_path , enkf_state->eclbase , ecl_restart_file , fmt_file , report_step);

  fortio_type * fortio = fortio_open(restart_file , "r" , endian_swap);
  
  if (unified)
    ecl_block_fseek(report_step , fmt_file , true , fortio);
  
  ecl_block = ecl_block_alloc(report_step , fmt_file , endian_swap);
  ecl_block_fread(ecl_block , fortio , &at_eof);
  fortio_close(fortio);
  
  enkf_state_load_ecl_restart__(enkf_state , ecl_block);
  ecl_block_free(ecl_block);
  free(restart_file);
}





void enkf_state_load_ecl_summary(enkf_state_type * enkf_state, bool unified , int report_step) {
  const bool fmt_file = enkf_state_fmt_file(enkf_state);
  ecl_sum_type * ecl_sum;
  int Nwells;
  const char ** well_list = enkf_config_get_well_list_ref(enkf_state->config , &Nwells);
  char * summary_file     = ecl_util_alloc_exfilename(enkf_state->run_path , enkf_state->eclbase , ecl_summary_file        , fmt_file ,  report_step);
  char * header_file      = ecl_util_alloc_exfilename(enkf_state->run_path , enkf_state->eclbase , ecl_summary_header_file , fmt_file , -1);

  int iwell;
  ecl_sum = ecl_sum_fread_alloc(header_file , 1 , (const char **) &summary_file , true , enkf_config_get_endian_swap(enkf_state->config));
  for (iwell = 0; iwell < Nwells; iwell++) {
    if (! enkf_state_has_node(enkf_state , well_list[iwell])) 
      enkf_state_add_node__1(enkf_state , well_list[iwell] , enkf_config_get_ref(enkf_state->config , well_list[iwell])); 
    {
      enkf_node_type * enkf_node = enkf_state_get_node(enkf_state , well_list[iwell]);
      well_load_summary_data(enkf_node_value_ptr(enkf_node) , report_step , ecl_sum);
    }
  }
  ecl_sum_free(ecl_sum);
  free(summary_file);
  free(header_file);
}


void enkf_state_load_ecl(enkf_state_type * enkf_state , bool unified , int report_step) {
  enkf_state_load_ecl_restart(enkf_state , unified , report_step);
  enkf_state_load_ecl_summary(enkf_state , unified , report_step);
}


void * enkf_state_load_ecl_summary_void(void * input_arg) {
  void_arg_type * arg = (void_arg_type *) input_arg;
  enkf_state_type * enkf_state;
  bool unified;
  int report_step;
  
  enkf_state  = void_arg_get_ptr(arg , 0);
  unified     = void_arg_get_bool(arg , 1 );
  report_step = void_arg_get_int(arg , 2 );

  enkf_state_load_ecl_summary(enkf_state , unified , report_step);
  return NULL;
}


void * enkf_state_load_ecl_restart_void(void * input_arg) {
  void_arg_type * arg = (void_arg_type *) input_arg;
  enkf_state_type * enkf_state;
  bool unified;
  int report_step;
  
  enkf_state  = void_arg_get_ptr(arg , 0);
  unified     = void_arg_get_bool(arg , 1 );
  report_step = void_arg_get_int(arg , 2 );

  enkf_state_load_ecl_restart(enkf_state , unified , report_step);
  return NULL;
}


void * enkf_state_load_ecl_void(void * input_arg) {
  void_arg_type * arg = (void_arg_type *) input_arg;
  enkf_state_type * enkf_state;
  bool unified;
  int report_step;
  
  enkf_state  = void_arg_get_ptr(arg , 0);
  unified     = void_arg_get_bool(arg , 1 );
  report_step = void_arg_get_int(arg , 2 );
  enkf_state_load_ecl(enkf_state , unified , report_step);
  return NULL;
}



void enkf_state_ecl_write(const enkf_state_type * enkf_state ,  int mask , int report_step) {
  const int buffer_size = 65536;
  const bool fmt_file   = enkf_state_fmt_file(enkf_state);
  bool endian_swap      = enkf_config_get_endian_swap(enkf_state->config);
  char  * restart_file  = ecl_util_alloc_filename(enkf_state->run_path , enkf_state->eclbase , ecl_restart_file , fmt_file , report_step);
  fortio_type * fortio  = fortio_open(restart_file , "w" , endian_swap);
  void *buffer          = malloc(buffer_size);
  list_node_type *list_node;                                            

  util_make_path(enkf_state->run_path);
  list_node = list_get_head(enkf_state->node_list);                     
  while (list_node != NULL) {                                           
    enkf_node_type * enkf_node = list_node_value_ptr(list_node);         

    if (enkf_node_include_type(enkf_node , mask)) {
      bool swapped = enkf_node_swapped(enkf_node);
      {
	bool analyzed = true;
	if (swapped) enkf_fs_swapin_node(enkf_state->enkf_fs , enkf_node , report_step , enkf_state->my_iens , analyzed);
      }

      if (enkf_node_include_type(enkf_node , ecl_restart)) {      
	/* Pressure and saturations */
	if (enkf_node_get_impl_type(enkf_node) == FIELD)
	  field_ecl_write1D_fortio(enkf_node_value_ptr(enkf_node) , fortio , fmt_file , endian_swap);
	else {
	  fprintf(stderr,"%s: internal error wrong implementetion type:%d - node:%s aborting \n",__func__ , enkf_node_get_impl_type(enkf_node) , enkf_node_get_key_ref(enkf_node));
	  abort();
	}	  
      } else if (enkf_node_include_type(enkf_node , ecl_static)) {

	ecl_kw_fwrite(ecl_static_kw_ecl_kw_ptr((const ecl_static_kw_type *) enkf_node_value_ptr(enkf_node)) , fortio);
      } else if (enkf_node_include_type(enkf_node , parameter + static_parameter))
        enkf_node_ecl_write(enkf_node , enkf_state->run_path);
      
      {
	bool analyzed = true;
	if (swapped) enkf_fs_swapout_node(enkf_state->enkf_fs , enkf_node , report_step , enkf_state->my_iens , analyzed);
      }
    }
    list_node = list_node_get_next(list_node);
  }
  free(restart_file);
  fortio_close(fortio);
  free(buffer);
}




void enkf_state_ens_write(const enkf_state_type * enkf_state , int mask) {
  const char *path = NULL ; /*enkf_state_get_enspath_ref(enkf_state);*/
  list_node_type *list_node;                                            
  list_node  = list_get_head(enkf_state->node_list);                    
  while (list_node != NULL) {                                           
    enkf_node_type *enkf_node = (enkf_node_type *) list_node_value_ptr(list_node);        
    if (enkf_node_include_type(enkf_node , mask))                       
      /*enkf_node_fwrite(enkf_node , path);                       */
    list_node  = list_node_get_next(list_node);                         
  }                                                                     
}


void enkf_state_swapout(enkf_state_type * enkf_state , int mask , bool forecast) {
  list_node_type *list_node;                                            
  list_node  = list_get_head(enkf_state->node_list);                    
  while (list_node != NULL) {                                           
    enkf_node_type *enkf_node = list_node_value_ptr(list_node);        
    if (enkf_node_include_type(enkf_node , mask))                       
      enkf_node_swapout(enkf_node , enkf_state_select_ens_path(enkf_state , enkf_node_get_key_ref(enkf_node) , forecast));
    list_node  = list_node_get_next(list_node);                         
  }                                                                     
}


void enkf_state_swapin(enkf_state_type * enkf_state , int mask , bool forecast) {
  list_node_type *list_node;                                            
  list_node  = list_get_head(enkf_state->node_list);                    
  while (list_node != NULL) {                                           
    enkf_node_type *enkf_node = list_node_value_ptr(list_node);        
    if (enkf_node_include_type(enkf_node , mask))                       
      enkf_node_swapin(enkf_node , enkf_state_select_ens_path(enkf_state , enkf_node_get_key_ref(enkf_node) , forecast));
    list_node  = list_node_get_next(list_node);                         
  }                                                                     
}



void enkf_state_free_nodes(enkf_state_type * enkf_state, int mask) {
  list_node_type *list_node;                                            
  list_node = list_get_head(enkf_state->node_list);                     
  while (list_node != NULL) {                                           
    list_node_type * next_node = list_node_get_next(list_node);
    enkf_node_type * enkf_node = list_node_value_ptr(list_node);         
    
    if (enkf_node_include_type(enkf_node , mask))      
      enkf_state_del_node(enkf_state , enkf_node_get_key_ref(enkf_node));
    
    list_node = next_node;
  } 
}


/*
void enkf_state_serialize(enkf_state_type * enkf_state , size_t stride) {
  {
    list_node_type *list_node;                                            
    list_node  = list_get_head(enkf_state->node_list);                    
    size_t offset = 0;
    size_t serial_data_size = 100;
    while (list_node != NULL) {                                           
      enkf_node_type *enkf_node = list_node_value_ptr(list_node);        
      if (enkf_node_include_type(enkf_node , parameter + ecl_restart + ecl_summary))
	offset += stride * enkf_node_serialize(enkf_node , serial_data_size , enkf_state->serial_data , stride , offset);                       
      list_node  = list_node_get_next(list_node);                         
    }             
  }
}
*/


void enkf_state_free(enkf_state_type *enkf_state) {
  list_free(enkf_state->node_list);
  hash_free(enkf_state->node_hash);
  hash_free(enkf_state->impl_types);
  free(enkf_state->run_path);
  enkf_fs_free(enkf_state->enkf_fs);
  restart_kw_list_free(enkf_state->restart_kw_list);
  free(enkf_state->eclbase);
  free(enkf_state);
}



enkf_node_type * enkf_state_get_node(const enkf_state_type * enkf_state , const char * node_key) {
  if (hash_has_key(enkf_state->node_hash , node_key)) {
    list_node_type * list_node = hash_get(enkf_state->node_hash , node_key);
    enkf_node_type * enkf_node = list_node_value_ptr(list_node);
    return enkf_node;
  } else {
    fprintf(stderr,"%s: node:%s not found in state object - aborting \n",__func__ , node_key);
    abort();
  }
}



void enkf_state_del_node(enkf_state_type * enkf_state , const char * node_key) {
  if (hash_has_key(enkf_state->node_hash , node_key)) {
    list_node_type * list_node = hash_get(enkf_state->node_hash , node_key);
    
    hash_del(enkf_state->node_hash , node_key);
    list_del_node(enkf_state->node_list , list_node);
    
  } else {
    fprintf(stderr,"%s: node:%s not found in state object - aborting \n",__func__ , node_key);
    abort();
  } 
}




static double * enkf_ensemble_alloc_serial_data(int ens_size , size_t target_serial_size , size_t * _serial_size) {
  size_t   serial_size = target_serial_size / ens_size;
  double * serial_data;
  do {
    serial_data = malloc(serial_size * ens_size * sizeof * serial_data);
    if (serial_data == NULL) 
      serial_size /= 2;
  } while (serial_data == NULL);
  *_serial_size = serial_size * ens_size;
  return serial_data;
}




void enkf_ensemble_mulX(double * serial_state , int serial_x_stride , int serial_y_stride , int serial_x_size , int serial_y_size , const double * X , int X_x_stride , int X_y_stride) {
  double * line = malloc(serial_x_size * sizeof * line);
  int ix,iy;
  
  for (iy=0; iy < serial_y_size; iy++) {
    if (serial_x_stride == 1) 
      memcpy(line , &serial_state[iy * serial_y_stride] , serial_x_size * sizeof * line);
    else
      for (ix = 0; ix < serial_x_size; ix++)
	line[ix] = serial_state[iy * serial_y_stride + ix * serial_x_stride];

    for (ix = 0; ix < serial_x_size; ix++) {
      int k;
      double dot_product = 0;
      for (k = 0; k < serial_x_size; k++)
	dot_product += line[k] * X[ix * X_x_stride + k*X_y_stride];
      serial_state[ix * serial_x_stride + iy * serial_y_stride] = dot_product;
    }

  }
  
  free(line);
}



void * enkf_ensemble_serialize_threaded(void * _void_arg) {
  void_arg_type * void_arg     = ( void_arg_type * ) _void_arg;
  int update_mask;
  int iens , iens1 , iens2 , serial_stride;
  size_t serial_size;
  double *serial_data;
  size_t * member_serial_size;
  bool   * member_complete;
  list_node_type ** start_node; 
  list_node_type ** next_node;  
  
  iens1       	     = void_arg_get_int(void_arg , 0 );
  iens2       	     = void_arg_get_int(void_arg , 1 );
  serial_size 	     = void_arg_get_size_t(void_arg , 2);
  serial_stride      = void_arg_get_int(void_arg , 3);
  serial_data        = void_arg_get_ptr(void_arg , 4);
  start_node 	     = void_arg_get_ptr(void_arg , 5);
  next_node  	     = void_arg_get_ptr(void_arg , 6);
  member_serial_size = void_arg_get_ptr(void_arg , 7);
  member_complete    = void_arg_get_ptr(void_arg , 8);
  update_mask        = void_arg_get_int(void_arg , 9);
  for (iens = iens1; iens < iens2; iens++) {
    list_node_type  * list_node  = start_node[iens];
    bool node_complete           = true;  
    size_t   serial_offset       = iens;
    
    while (node_complete) {                                           
      enkf_node_type *enkf_node = list_node_value_ptr(list_node);        
      if (enkf_node_include_type(enkf_node , update_mask)) {                       
	int elements_added = enkf_node_serialize(enkf_node , serial_size , serial_data , serial_stride , serial_offset , &node_complete);
	serial_offset            += serial_stride * elements_added;  
	member_serial_size[iens] += elements_added;
      }
      
      if (node_complete) {
	list_node  = list_node_get_next(list_node);                         
	if (list_node == NULL) {
	  if (node_complete) member_complete[iens] = true;
	  break;
	}
      }
    }
      /* Restart on this node */
    next_node[iens] = list_node;
  }
  
  return NULL;
}



void enkf_ensemble_update(enkf_state_type ** enkf_ens , int ens_size , size_t target_serial_size , const double * X) {
  const int threads = 4;
  int update_mask = ecl_summary + ecl_restart + parameter;
  thread_pool_type * tp = thread_pool_alloc(threads);
  void_arg_type ** void_arg    = malloc(threads * sizeof * void_arg);
  int *     iens1              = malloc(threads * sizeof * iens1);
  int *     iens2              = malloc(threads * sizeof * iens2);
  bool *    member_complete    = malloc(ens_size * sizeof * member_complete);
  size_t  * member_serial_size = malloc(ens_size * sizeof * member_serial_size);
  size_t    serial_size;
  double *  serial_data   = enkf_ensemble_alloc_serial_data(ens_size , target_serial_size , &serial_size);
  int       serial_stride = ens_size;
  int       iens , ithread;
  

  bool      state_complete = false;
  list_node_type  ** start_node = malloc(ens_size * sizeof * start_node);
  list_node_type  ** next_node  = malloc(ens_size * sizeof * next_node);
  
  
  for (iens = 0; iens < ens_size; iens++) {
    enkf_state_type * enkf_state = enkf_ens[iens];
    start_node[iens] = list_get_head(enkf_state->node_list);                    
    enkf_state_apply(enkf_ens[iens] , enkf_node_clear_serial_state , update_mask);
    member_complete[iens] = false;
  }
  
  {
    int thread_block_size = ens_size / threads;
    for (ithread = 0; ithread < threads; ithread++) {
      iens1[ithread] = ithread * thread_block_size;
      iens2[ithread] = iens1[ithread] + thread_block_size;
      
      void_arg[ithread] = void_arg_alloc10(int_value 	 ,     /* 0 */
					   int_value 	 ,     /* 1 */
					   size_t_value  ,     /* 2 */
					   int_value     ,     /* 3 */
					   pointer_value ,     /* 4 */
					   pointer_value ,     /* 5 */
					   pointer_value ,     /* 6 */
					   pointer_value ,     /* 7 */
					   pointer_value ,     /* 8 */ 
					   int_value       );  /* 9 */
    }
    iens2[threads-1] = ens_size;
  }


  while (!state_complete) {
    for (iens = 0; iens < ens_size; iens++) 
      member_serial_size[iens] = 0;
    
    for (ithread =  0; ithread < threads; ithread++) {
      void_arg_pack_int(void_arg[ithread]     , 0 , iens1[ithread]);
      void_arg_pack_int(void_arg[ithread]     , 1 , iens2[ithread]);
      void_arg_pack_size_t(void_arg[ithread]  , 2 , serial_size);
      void_arg_pack_int(void_arg[ithread]     , 3 , serial_stride);
      void_arg_pack_ptr(void_arg[ithread]     , 4 , serial_data);
      void_arg_pack_ptr(void_arg[ithread]     , 5 , start_node);
      void_arg_pack_ptr(void_arg[ithread]     , 6 , next_node);
      void_arg_pack_ptr(void_arg[ithread]     , 7 , member_serial_size);
      void_arg_pack_ptr(void_arg[ithread]     , 8 , member_complete);
      void_arg_pack_int(void_arg[ithread]    , 9 , update_mask);
    }
    
    
    for (ithread =  0; ithread < threads; ithread++) 
      thread_pool_add_job(tp , &enkf_ensemble_serialize_threaded , void_arg[ithread]);
    thread_pool_join(tp);
    

    /* Serialize section */
/*     for (iens = 0; iens < ens_size; iens++) { */
/*       list_node_type  * list_node  = start_node[iens]; */
/*       bool node_complete           = true;   */
/*       size_t   serial_offset       = iens; */
      
/*       while (node_complete) {                                            */
/* 	enkf_node_type *enkf_node = list_node_value_ptr(list_node);         */
/* 	if (enkf_node_include_type(enkf_node , update_mask)) {                        */
/* 	  int elements_added = enkf_node_serialize(enkf_node , serial_size , serial_data , serial_stride , serial_offset , &node_complete); */
/* 	  serial_offset            += serial_stride * elements_added;   */
/* 	  member_serial_size[iens] += elements_added; */
/* 	} */
	
/* 	if (node_complete) { */
/* 	  list_node  = list_node_get_next(list_node);                          */
/* 	  if (list_node == NULL) { */
/* 	    if (node_complete) member_complete[iens] = true; */
/* 	    break; */
/* 	  } */
/* 	} */
/*       } */
/*       /\* Restart on this node *\/ */
/*       next_node[iens] = list_node; */
/*     } */

    for (iens=1; iens < ens_size; iens++) {
      if (member_complete[iens]    != member_complete[iens-1])    {  fprintf(stderr,"%s: member_complete difference    - INTERNAL ERROR - aborting \n",__func__); abort(); }
      if (member_serial_size[iens] != member_serial_size[iens-1]) {  fprintf(stderr,"%s: member_serial_size difference - INTERNAL ERROR - aborting \n",__func__); abort(); }
    }
    state_complete = member_complete[0];
    
    

    /* Update section */
    enkf_ensemble_mulX(serial_data , 1 , ens_size , ens_size , member_serial_size[0] , X , ens_size , 1);

    /* deserialize section */
    for (iens = 0; iens < ens_size; iens++) {
      list_node_type  * list_node  = start_node[iens];
      
      while (1) {
	enkf_node_type *enkf_node = list_node_value_ptr(list_node);        
	if (enkf_node_include_type(enkf_node , update_mask)) 
	  enkf_node_deserialize(enkf_node , serial_data , serial_stride);
	
	if (list_node == next_node[iens])
	  break;
	
	list_node  = list_node_get_next(list_node);                         
	if (list_node == NULL)
	  break;
      }
    }
    
    for (iens = 0; iens < ens_size; iens++) 
      start_node[iens] = next_node[iens];
  }
  for (ithread = 0; ithread < threads; ithread++) 
    void_arg_free(void_arg[ithread]);
  thread_pool_free(tp);

  free(void_arg);
  free(member_complete);
  free(member_serial_size);
  free(iens1);
  free(iens2);


  free(start_node);
  free(serial_data);
  free(next_node);
}



/*****************************************************************/
/* Generatad functions - iterating through all members.          */
/*****************************************************************/


/*ENKF_STATE_APPLY_PATH(fread);*/
ENKF_STATE_APPLY(sample);
ENKF_STATE_APPLY(clear);
ENKF_STATE_APPLY(clear_serial_state);
ENKF_STATE_APPLY_SCALAR(scale);
ENKF_STATE_APPLY2(imul);
ENKF_STATE_APPLY2(iadd);
ENKF_STATE_APPLY2(iaddsqr);
