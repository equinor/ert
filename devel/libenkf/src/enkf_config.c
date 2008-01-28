#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <util.h>
#include <hash.h>
#include <multz_ens.h>
#include <enkf_ens.h>
#include <enkf_ens_node.h>
#include <path_fmt.h>
#include <ecl_static_kw_ens.h>
#include <enkf_types.h>
#include <well_ens.h>
#include <field_ens.h>
#include <equil_ens.h>
#include <multz_ens.h>
#include <multflt_ens.h>
#include <well_obs.h>
#include <pgbox_ens.h>
#include <obs_node.h>
#include <obs_data.h>
#include <history.h>
  

struct enkf_ens_struct {
  int  		    ens_size;
  int               Nwells;
  bool              endian_swap;
  hash_type        *ens_hash;
  hash_type        *obs_hash;
  path_fmt_type    *run_path;
  char            **well_list;
};


/*****************************************************************/




enkf_impl_type enkf_ens_impl_type(const enkf_ens_type *enkf_ens, const char * ecl_kw_name) {
  enkf_impl_type impl_type;

  if (hash_has_key(enkf_ens->ens_hash , ecl_kw_name)) {
    enkf_ens_node_type * node = hash_get(enkf_ens->ens_hash , ecl_kw_name);
    impl_type = enkf_ens_node_get_impl_type(node);
  } else
    impl_type = STATIC;

  return impl_type;
}



/*****************************************************************/
static void enkf_ens_realloc_well_list(enkf_ens_type * enkf_ens) {
  enkf_ens->well_list = realloc(enkf_ens->well_list , enkf_ens->Nwells * sizeof * enkf_ens->well_list);
}

bool enkf_ens_get_endian_swap(const enkf_ens_type * enkf_ens) { return enkf_ens->endian_swap; }


enkf_ens_type * enkf_ens_alloc(const char * run_path , const char * ens_path_static , const char * ens_path_parameter , const char * ens_path_dynamic_forecast , const char * ens_path_dynamic_analyzed , bool endian_swap) {

  enkf_ens_type * enkf_ens = malloc(sizeof *enkf_ens);
  enkf_ens->ens_hash = hash_alloc(10);
  enkf_ens->obs_hash    = hash_alloc(10);
  
  enkf_ens->endian_swap   = endian_swap;
  enkf_ens->Nwells        = 0;
  enkf_ens->well_list     = NULL;
  enkf_ens_realloc_well_list(enkf_ens);
  
  enkf_ens->run_path           	    = path_fmt_alloc_directory_fmt(run_path , true);
  /*
  enkf_ens->ens_path_parameter 	    = path_fmt_alloc_directory_fmt(ens_path_parameter , true);
  enkf_ens->ens_path_static    	    = path_fmt_alloc_directory_fmt(ens_path_static , true);
  enkf_ens->ens_path_dynamic_forecast    = path_fmt_alloc_directory_fmt(ens_path_dynamic_forecast , true);
  enkf_ens->ens_path_dynamic_analyzed    = path_fmt_alloc_directory_fmt(ens_path_dynamic_analyzed , true);
  */
  return enkf_ens;
}



bool enkf_ens_has_key(const enkf_ens_type * enkf_ens , const char * key) {
  return hash_has_key(enkf_ens->ens_hash , key);
}


const char ** enkf_ens_get_well_list_ref(const enkf_ens_type * ens , int *Nwells) {
  *Nwells = ens->Nwells;
  return (const char **) ens->well_list;
}


void enkf_ens_add_well(enkf_ens_type * enkf_ens , const char *well_name , int size, const char ** var_list) {
  enkf_ens_add_type(enkf_ens , well_name , ecl_summary , WELL,
		       well_ens_alloc(well_name , size , var_list));
  
  enkf_ens->Nwells++;
  enkf_ens_realloc_well_list(enkf_ens);
  enkf_ens->well_list[enkf_ens->Nwells - 1] = util_alloc_string_copy(well_name);
}




void enkf_ens_add_type(enkf_ens_type * enkf_ens, 
			  const char * key , 
			  enkf_var_type enkf_type , 
			  enkf_impl_type impl_type , 
			  const void *data) {
  if (enkf_ens_has_key(enkf_ens , key)) {
    fprintf(stderr,"%s: a ensuration object:%s has already been added - aborting \n",__func__ , key);
    abort();
  }

  {
    ens_free_ftype              * freef;
    switch(impl_type) {
    case(FIELD):
      freef             = field_ens_free__;
      break;
    case(MULTZ):
      freef             = multz_ens_free__;
      break;
    case(WELL):
      freef             = well_ens_free__;
      break;
    case(MULTFLT):
      freef             = multflt_ens_free__;
      break;
    case(EQUIL):
      freef             = equil_ens_free__;
      break;
    case(STATIC):
      freef             = ecl_static_kw_ens_free__;
      break;
    case(PGBOX):
      freef             = pgbox_ens_free__;
      break;
    default:
      fprintf(stderr,"%s : invalid implementation type: %d - aborting \n",__func__ , impl_type);
      abort();
    }
    {
      enkf_ens_node_type * node = enkf_ens_node_alloc(enkf_type , impl_type , key , NULL , data , freef);
      hash_insert_hash_owned_ref(enkf_ens->ens_hash , key , node , enkf_ens_node_free__);
    }
  }
}



void enkf_ens_add_type0(enkf_ens_type * enkf_ens , const char *key , int size, enkf_var_type enkf_type , enkf_impl_type impl_type) {
  switch(impl_type) {
  case(STATIC):
    enkf_ens_add_type(enkf_ens , key , enkf_type , impl_type , ecl_static_kw_ens_alloc(size , key , key));
    break;
  case(FIELD):
    /*
      enkf_ens_add_type(enkf_ens , key , enkf_type , impl_type , field_ens_alloc(size , key , key)   , field_ens_free__ , field_ens_get_size__);
    */
    fprintf(stderr,"%s: Can not add FIELD ens objects like:%s on the run - these must be from the main program with enkf_ens_add_type - sorry.\n",__func__ , key);
    abort();
    break;
  default:
    fprintf(stderr,"%s only STATIC and FIELD types are implemented - aborting \n",__func__);
    abort();
  }
}



void enkf_ens_free(enkf_ens_type * enkf_ens) {  
  hash_free(enkf_ens->ens_hash);
  hash_free(enkf_ens->obs_hash);
  {
    int i;
    for (i=0; i < enkf_ens->Nwells; i++)
      free(enkf_ens->well_list[i]);
    free(enkf_ens->well_list);
  }
  path_fmt_free(enkf_ens->run_path);
  /*
    path_fmt_free(enkf_ens->ens_path_parameter);
    path_fmt_free(enkf_ens->ens_path_static);
    path_fmt_free(enkf_ens->ens_path_dynamic_forecast);
    path_fmt_free(enkf_ens->ens_path_dynamic_analyzed);
  */
  free(enkf_ens);
}



const enkf_ens_node_type * enkf_ens_get_ref(const enkf_ens_type * ens, const char * key) {
  if (hash_has_key(ens->ens_hash , key)) {
    enkf_ens_node_type * node = hash_get(ens->ens_hash , key);
    return node;
  } else {
    fprintf(stderr,"%s: ens node:%s does not exist \n",__func__ , key);
    abort();
  }
}


const path_fmt_type * enkf_ens_get_run_path_ref(const enkf_ens_type *ens) { return ens->run_path; }





