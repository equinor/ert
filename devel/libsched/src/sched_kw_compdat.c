#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sched_kw_compdat.h>
#include <list.h>
#include <util.h>
#include <list_node.h>
#include <sched_util.h>
#include <stdbool.h>

static  const double Kmin = 0.10;

typedef enum {X , Y , Z}  well_dir_type;       
typedef enum {OPEN , AUTO , SHUT}   comp_state_type;


struct sched_kw_compdat_struct {
  int        kw_size;
  list_type *comp_list;
};


typedef struct  {
  char             *well;
  char             *well_dir_string;
  char             *comp_string;

  int               i,j,k1,k2;
  comp_state_type   comp_state;
  int               sat_table;
  double            conn_factor;
  double            well_diameter;     
  double            eff_perm;	       
  double            skin_factor;       
  double            D_factor;	       
  well_dir_type     well_dir;	       
  double            r0;                   
  
  double            conn_factor__;
  bool             *def;
} comp_type;


/*****************************************************************/

/*
  Using the following simplified model for the well connection factors:

  1. Only the major well direction, X/Y/Z is considered, i.e. the
     detailed direction is ignored.

  2. The permeability dependence in the denominator is ignored, i.e

       Tx = G * Sqrt(Ky * Kz)
       Ty = G * Sqrt(Kx * Kz)
       Tz = G * Sqrt(Kx * Ky)

     Where G is unspesified constant.

  3. We assume Kx == Ky.
*/
  
  
static void comp_sched_init_conn_factor(comp_type * comp , const float *permx_field, const float * permz_field , const int * dims , const int * index_field , bool *OK) {
  const int i         = comp->i  - 1;
  const int j         = comp->j  - 1;
  const int k         = comp->k1 - 1;
  const int index_arg = i + j*dims[0] + k*dims[0] * dims[1];
  const int index     = index_field[index_arg] - 1;
  
  const double Kx   = permx_field[index];
  const double Ky   = permx_field[index];
  const double Kz   = permz_field[index];
  double Keff;
  
  switch (comp->well_dir) {
  case(X):
    Keff = sqrt(Ky * Kz);
    break;
  case(Y):
    Keff = sqrt(Kx * Kz);
    break;
  case(Z):
    Keff = sqrt(Kx * Ky);
    break;
  default:
    fprintf(stderr,"%s: comp->well_dir = %d - undefined INTERNAL ERROR - aborting \n", __func__ , comp->well_dir);
    abort();
  }
  if (Keff <= Kmin) {
    fprintf(stderr,"%s: In well:%s (cell: %d, %d, %d) the effective permeability: is zero - can *not* update connection factor for this cell. K = (%g,%g,%g), Cf = %g \n",__func__ , comp->well , (i+1) , (j+1) , (k+1),
	    Kx,Ky,Kz,comp->conn_factor);
    comp->conn_factor__ = -1;
    /* 
       A negative value for conn_factor__ means that the original value is used for all members,
       irrespective of permeability updates.
    */
  } else
    comp->conn_factor__ = comp->conn_factor / Keff;
}



static void comp_sched_set_conn_factor(comp_type * comp , const float *permx_field, const float *permz_field , const int * dims , const int * index_field) {
  if (comp->conn_factor__ > 0) {
    
    const int i         = comp->i  - 1;
    const int j         = comp->j  - 1;
    const int k         = comp->k1 - 1;
    const int index_arg = i + j*dims[0] + k*dims[0] * dims[1];
    const int index     = index_field[index_arg] - 1;
    const double Kx      = permx_field[index];
    const double Ky      = permx_field[index];
    const double Kz      = permz_field[index];
    
    double Keff = 0;

    switch (comp->well_dir) {
    case(X):
      Keff = sqrt(Ky * Kz);
      break;
    case(Y):
      Keff = sqrt(Kx * Kz);
      break;
    case(Z):
      Keff = sqrt(Kx * Ky);
      break;
    default:
      fprintf(stderr,"%s: internal error: variable comp->well_dir has an invalid value - aborting \n",__func__);
      abort();
    }
    comp->conn_factor = comp->conn_factor__ * Keff;
  }
}





static void comp_sched_fprintf(const comp_type * comp , FILE *stream) {
  fprintf(stream , " ");
  sched_util_fprintf_qst(comp->def[0] , comp->well 	      , 8  , stream);
  sched_util_fprintf_int(comp->def[1] , comp->i    	      , 4  , stream);
  sched_util_fprintf_int(comp->def[2] , comp->j    	      , 4  , stream);
  sched_util_fprintf_int(comp->def[3] , comp->k1   	      , 4  , stream);
  sched_util_fprintf_int(comp->def[4] , comp->k2   	      , 4  , stream);
  sched_util_fprintf_qst(comp->def[5] , comp->comp_string     , 4  , stream);
  sched_util_fprintf_int(comp->def[6] , comp->sat_table       , 6  ,     stream);
  sched_util_fprintf_dbl(comp->def[7] , comp->conn_factor     , 9  , 3 , stream);
  sched_util_fprintf_dbl(comp->def[8] , comp->well_diameter   , 9  , 3 , stream);
  sched_util_fprintf_dbl(comp->def[9] , comp->eff_perm        , 9  , 3 , stream);
  sched_util_fprintf_dbl(comp->def[10], comp->skin_factor     , 9  , 3 , stream);
  sched_util_fprintf_dbl(comp->def[11], comp->D_factor        , 9  , 3 , stream);
  sched_util_fprintf_qst(comp->def[12], comp->well_dir_string , 2  , stream);
  sched_util_fprintf_dbl(comp->def[13], comp->r0              , 9  , 3 , stream);
  fprintf(stream , " /\n");
}



static void comp_set_from_string(comp_type * node , int kw_size , const char **token_list ) {
  {
    int i;
    for (i=0; i < kw_size; i++) {
      if (token_list[i] == NULL)
	node->def[i] = true;
      else
	node->def[i] = false;
    }
  }


  node->well         = util_alloc_string_copy(token_list[0]);
  node->i            = sched_util_atoi(token_list[1]);
  node->j            = sched_util_atoi(token_list[2]);
  node->k1           = sched_util_atoi(token_list[3]);
  node->k2           = sched_util_atoi(token_list[4]);
  if (node->def[5]) {
    node->comp_string = malloc( 5 );
    sprintf(node->comp_string , "OPEN");
  } else
    node->comp_string  = util_alloc_string_copy(token_list[5]);
  
  if (strcmp(node->comp_string , "OPEN") == 0) node->comp_state = OPEN;
  if (strcmp(node->comp_string , "AUTO") == 0) node->comp_state = AUTO;
  if (strcmp(node->comp_string , "SHUT") == 0) node->comp_state = SHUT;
  
  node->sat_table       = sched_util_atoi(token_list[6]);
  node->conn_factor     = sched_util_atof(token_list[7]);
  node->well_diameter   = sched_util_atof(token_list[8]);     
  node->eff_perm        = sched_util_atof(token_list[9]);	       
  node->skin_factor     = sched_util_atof(token_list[10]);       
  node->D_factor        = sched_util_atof(token_list[11]);	       
  if (node->def[12]) {
    node->well_dir_string = malloc(2);
    sprintf(node->well_dir_string , "Z");
  } else 
    node->well_dir_string = util_alloc_string_copy(token_list[12]);
  
  node->well_dir = -1;
  if (strcmp(node->well_dir_string , "X")  == 0) node->well_dir = X;
  if (strcmp(node->well_dir_string , "Y")  == 0) node->well_dir = Y;
  if (strcmp(node->well_dir_string , "Z")  == 0) node->well_dir = Z;
  if (node->well_dir == -1) {
    fprintf(stderr,"%s: well_dir_string = %s not recognized - aborting \n",__func__ , node->well_dir_string);
    abort();
  }
  
  node->r0 = sched_util_atof(token_list[13]);                
}


static comp_type * comp_alloc_empty(int kw_size) {
  comp_type *node = malloc(sizeof *node);
  node->well          = NULL;
  node->conn_factor__ = -1;
  node->def           = calloc(kw_size , sizeof *node->def);
  return node;
}


static comp_type * comp_alloc(int kw_size , const char **token_list) {
  comp_type * node = comp_alloc_empty(kw_size);
  comp_set_from_string(node , kw_size , token_list);
  return node;
}


static void comp_free(comp_type *comp) {
  free(comp->well);
  free(comp->def);
  free(comp->well_dir_string);
  free(comp->comp_string);
  free(comp);
}


static void comp_free__(void *__comp) {
  comp_type *comp = (comp_type *) __comp;
  comp_free(comp);
}


static void comp_sched_fwrite(const comp_type *comp , int kw_size , FILE *stream) {
  util_fwrite_string(comp->well            , stream);
  util_fwrite_string(comp->comp_string     , stream);
  util_fwrite_string(comp->well_dir_string , stream);

  util_fwrite(&comp->i  	   , sizeof comp->i  	       	, 1 	  , stream , __func__);
  util_fwrite(&comp->j  	   , sizeof comp->j  	       	, 1 	  , stream , __func__);
  util_fwrite(&comp->k1 	   , sizeof comp->k1 	       	, 1 	  , stream , __func__);
  util_fwrite(&comp->k2 	   , sizeof comp->k2 	       	, 1 	  , stream , __func__);
  util_fwrite(&comp->sat_table     , sizeof comp->sat_table    	, 1 	  , stream , __func__);
  util_fwrite(&comp->conn_factor   , sizeof comp->conn_factor  	, 1 	  , stream , __func__);
  util_fwrite(&comp->well_diameter , sizeof comp->well_diameter	, 1 	  , stream , __func__);
  util_fwrite(&comp->eff_perm      , sizeof comp->eff_perm	, 1 	  , stream , __func__);
  util_fwrite(&comp->skin_factor   , sizeof comp->skin_factor   , 1 	  , stream , __func__);
  util_fwrite(&comp->D_factor      , sizeof comp->D_factor	, 1 	  , stream , __func__);
  util_fwrite(&comp->well_dir      , sizeof comp->well_dir      , 1 	  , stream , __func__);
  util_fwrite(&comp->r0            , sizeof comp->r0            , 1 	  , stream , __func__);
  util_fwrite(&comp->conn_factor__ , sizeof comp->conn_factor__ , 1 	  , stream , __func__);
  util_fwrite(comp->def            , sizeof * comp->def         , kw_size , stream , __func__);
}


static comp_type * comp_sched_fread_alloc(int kw_size , FILE * stream) {
  comp_type * comp = comp_alloc_empty(kw_size);
  comp->well        	= util_fread_alloc_string( stream );
  comp->comp_string 	= util_fread_alloc_string( stream );
  comp->well_dir_string = util_fread_alloc_string( stream );

  util_fread(&comp->i  	      	   , sizeof comp->i  	     	  , 1 	    , stream , __func__);
  util_fread(&comp->j  	      	   , sizeof comp->j  	     	  , 1 	    , stream , __func__);
  util_fread(&comp->k1 	      	   , sizeof comp->k1 	     	  , 1 	    , stream , __func__);
  util_fread(&comp->k2 	      	   , sizeof comp->k2 	     	  , 1 	    , stream , __func__);
  util_fread(&comp->sat_table      , sizeof comp->sat_table       , 1 	    , stream , __func__);
  util_fread(&comp->conn_factor    , sizeof comp->conn_factor     , 1 	    , stream , __func__);
  util_fread(&comp->well_diameter  , sizeof comp->well_diameter   , 1 	    , stream , __func__);
  util_fread(&comp->eff_perm       , sizeof comp->eff_perm	  , 1 	    , stream , __func__);
  util_fread(&comp->skin_factor    , sizeof comp->skin_factor     , 1 	    , stream , __func__);
  util_fread(&comp->D_factor       , sizeof comp->D_factor	  , 1 	    , stream , __func__);
  util_fread(&comp->well_dir       , sizeof comp->well_dir        , 1 	    , stream , __func__);
  util_fread(&comp->r0             , sizeof comp->r0              , 1 	    , stream , __func__);
  util_fread(&comp->conn_factor__  , sizeof comp->conn_factor__   , 1 	    , stream , __func__);
  util_fread(comp->def             , sizeof * comp->def           , kw_size , stream , __func__);
    
  return comp;
}

/*****************************************************************/



void sched_kw_compdat_init_conn_factor(sched_kw_compdat_type * kw , const ecl_kw_type *permx_kw, const ecl_kw_type * permz_kw , const int * dims , const int * index_field , bool *OK) {
  float *permx = ecl_kw_get_float_ptr(permx_kw);
  float *permz = ecl_kw_get_float_ptr(permz_kw);
  list_node_type *comp_node = list_get_head(kw->comp_list);
  while (comp_node != NULL) {
    comp_type * comp = list_node_value_ptr(comp_node);
    comp_sched_init_conn_factor(comp , permx , permz , dims , index_field , OK);
    comp_node = list_node_get_next(comp_node);
  }
}


void sched_kw_compdat_set_conn_factor(sched_kw_compdat_type * kw , const float *permx , const float *permz , const int * dims , const int * index_field) {
  list_node_type *comp_node = list_get_head(kw->comp_list);
  while (comp_node != NULL) {
    comp_type * comp = list_node_value_ptr(comp_node);
    comp_sched_set_conn_factor(comp , permx , permz , dims , index_field);
    comp_node = list_node_get_next(comp_node);
  }
}



void sched_kw_compdat_fprintf(const sched_kw_compdat_type *kw , FILE *stream) {
  fprintf(stream , "COMPDAT\n");
  {
    list_node_type *comp_node = list_get_head(kw->comp_list);
    while (comp_node != NULL) {
      const comp_type * comp = list_node_value_ptr(comp_node);
      comp_sched_fprintf(comp , stream);
      comp_node = list_node_get_next(comp_node);
    }
  }
  fprintf(stream , "/\n\n");
}



sched_kw_compdat_type * sched_kw_compdat_alloc( ) {
  sched_kw_compdat_type * kw = malloc(sizeof *kw);
  kw->comp_list = list_alloc();
  kw->kw_size   = 14;
  return kw;
}


void sched_kw_compdat_add_line(sched_kw_compdat_type * kw , const char * line) {
  int tokens;
  char **token_list;
  
  sched_util_parse_line(line , &tokens , &token_list , kw->kw_size , NULL);
  {
    comp_type * comp = comp_alloc(kw->kw_size , (const char **) token_list);
    list_append_list_owned_ref(kw->comp_list , comp , comp_free__);
  }

  sched_util_free_token_list(tokens , token_list);
}


void sched_kw_compdat_free(sched_kw_compdat_type * kw) {
  list_free(kw->comp_list);
  free(kw);
}




void sched_kw_compdat_fwrite(const sched_kw_compdat_type *kw , FILE *stream) {
  util_fwrite(&kw->kw_size , sizeof kw->kw_size , 1 , stream , __func__);
  {
    int compdat_lines = list_get_size(kw->comp_list);
    util_fwrite(&compdat_lines , sizeof compdat_lines , 1, stream , __func__);
  }
  {
    list_node_type *comp_node = list_get_head(kw->comp_list);
    while (comp_node != NULL) {
      const comp_type * comp = list_node_value_ptr(comp_node);
      comp_sched_fwrite(comp , kw->kw_size , stream);
      comp_node = list_node_get_next(comp_node);
    }
  }
}



sched_kw_compdat_type * sched_kw_compdat_fread_alloc(FILE *stream) {
  sched_kw_compdat_type *kw = sched_kw_compdat_alloc();
  int lines , i;
  util_fread(&kw->kw_size , sizeof kw->kw_size , 1 , stream , __func__);
  util_fread(&lines       , sizeof lines       , 1 , stream , __func__);
  for (i=0; i < lines; i++) {
    comp_type * comp = comp_sched_fread_alloc(kw->kw_size , stream);
    list_append_list_owned_ref(kw->comp_list , comp , comp_free__);
  } 
  return kw;
}
  










