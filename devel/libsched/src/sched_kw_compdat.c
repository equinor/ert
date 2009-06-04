#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sched_kw_compdat.h>
#include <vector.h>
#include <util.h>
#include <sched_util.h>
#include <stdbool.h>

#define COMPDAT_NUM_KW        14   
#define SCHED_KW_COMPDAT_ID   771882

typedef enum {X, Y , Z , FX , FY}   well_dir_type; 
#define WELL_DIR_DEFAULT     Z
#define WELL_DIR_X_STRING   "X"
#define WELL_DIR_Y_STRING   "Y"
#define WELL_DIR_Z_STRING   "Z"
#define WELL_DIR_FX_STRING  "FX"
#define WELL_DIR_FY_STRING  "FZ"



typedef enum {OPEN , AUTO , SHUT}   comp_state_type;
#define COMP_DEFAULT_STATE  OPEN
#define COMP_OPEN_STRING   "OPEN"
#define COMP_AUTO_STRING   "AUTO"
#define COMP_SHUT_STRING   "SHUT"



/*
  Structure to hold one line (typically one completed cell) in a compdat section.
*/

typedef struct  {
  char             *well;            /* Name of well                                           */
  int               i,j,k1,k2;       /* The i,j,k coordinated of the perforated cell.          */
  well_dir_type     well_dir;        /* Which direction does the well penetrate the grid block */
  comp_state_type   state;           /* What state is this completion in: AUTO|SHUT|OPEN       */
  int               sat_table;    
  double            conn_factor;
  double            well_diameter;     
  double            eff_perm;	       
  double            skin_factor;       
  double            D_factor;	       
  double            r0;                   
  
  /*
    def : Read as defaulted, not as defined.
  */
  bool              def[COMPDAT_NUM_KW];
} comp_type;



struct sched_kw_compdat_struct {
  int            __type_id;
  vector_type  * completions;
};





static char * comp_get_state_string(comp_state_type state) {
  switch(state) {
  case(AUTO):
    return COMP_AUTO_STRING;
  case(OPEN):
    return COMP_OPEN_STRING;
  case(SHUT):
    return COMP_SHUT_STRING;
  default:
    util_abort("%s: internal error \n",__func__);
    return NULL;
  }
}



static char * comp_get_dir_string(well_dir_type dir) {
  switch(dir) {
  case(X):
    return WELL_DIR_X_STRING;
  case(Y):
    return WELL_DIR_Y_STRING;
  case(Z):
    return WELL_DIR_Z_STRING;
  case(FX):
    return WELL_DIR_FX_STRING;
  case(FY):
    return WELL_DIR_FY_STRING;
  default:
    util_abort("%s: internal fuckup \n",__func__);
    return NULL;
  }
}


static well_dir_type comp_get_well_dir_from_string(const char * well_dir) {
  if (strcmp(well_dir , WELL_DIR_X_STRING) == 0)
    return X;
  else if (strcmp(well_dir , WELL_DIR_Y_STRING) == 0)
    return Y;
  else if (strcmp(well_dir , WELL_DIR_Z_STRING) == 0)
    return X;
  else if (strcmp(well_dir , WELL_DIR_FX_STRING) == 0)
    return FX;
  else if (strcmp(well_dir , WELL_DIR_FY_STRING) == 0)
    return FY;
  else {
    util_abort("%s: internal fuckup \n",__func__);
    return -1;
  }
}



static comp_state_type comp_get_state_from_string(const char * state) {
  if (strcmp(state , COMP_AUTO_STRING) == 0)
    return AUTO;
  else if (strcmp(state , COMP_OPEN_STRING) == 0)
    return OPEN;
  else if (strcmp(state , COMP_SHUT_STRING) == 0)
    return SHUT;
  else {
    util_abort("%s: internal fuckup \n",__func__);
    return -1;
  }
}



static void comp_sched_fprintf(const comp_type * comp , FILE *stream) {
  fprintf(stream , " ");
  sched_util_fprintf_qst(comp->def[0] , comp->well 	      		     , 8  , stream);
  sched_util_fprintf_int(comp->def[1] , comp->i    	      		     , 4  , stream);
  sched_util_fprintf_int(comp->def[2] , comp->j    	      		     , 4  , stream);
  sched_util_fprintf_int(comp->def[3] , comp->k1   	      		     , 4  , stream);
  sched_util_fprintf_int(comp->def[4] , comp->k2   	      		     , 4  , stream);
  sched_util_fprintf_qst(comp->def[5] , comp_get_state_string( comp->state ) , 4  , stream);
  sched_util_fprintf_int(comp->def[6] , comp->sat_table       		     , 6  ,     stream);
  sched_util_fprintf_dbl(comp->def[7] , comp->conn_factor     		     , 9  , 3 , stream);
  sched_util_fprintf_dbl(comp->def[8] , comp->well_diameter   		     , 9  , 3 , stream);
  sched_util_fprintf_dbl(comp->def[9] , comp->eff_perm        		     , 9  , 3 , stream);
  sched_util_fprintf_dbl(comp->def[10], comp->skin_factor     		     , 9  , 3 , stream);
  sched_util_fprintf_dbl(comp->def[11], comp->D_factor        		     , 9  , 3 , stream);
  sched_util_fprintf_qst(comp->def[12], comp_get_dir_string( comp->well_dir) , 2  , stream);
  sched_util_fprintf_dbl(comp->def[13], comp->r0              		     , 9  , 3 , stream);
  fprintf(stream , " / -- Internal COMPDAT\n");
}



static void comp_set_from_string(comp_type * node , const char **token_list ) {
  {
    int i;
    for (i=0; i < COMPDAT_NUM_KW; i++) {
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

  if (node->def[5]) 
    node->state = COMP_DEFAULT_STATE;
  else 
    node->state = comp_get_state_from_string( token_list[5] );
  
  node->sat_table       = sched_util_atoi(token_list[6]);
  node->conn_factor     = sched_util_atof(token_list[7]);
  node->well_diameter   = sched_util_atof(token_list[8]);     
  node->eff_perm        = sched_util_atof(token_list[9]);	       
  node->skin_factor     = sched_util_atof(token_list[10]);       
  node->D_factor        = sched_util_atof(token_list[11]);	       
  if (node->def[12]) 
    node->well_dir = WELL_DIR_DEFAULT;
  else
    node->well_dir = comp_get_well_dir_from_string( token_list[12] );
  
  node->r0 = sched_util_atof(token_list[13]);                
}


static comp_type * comp_alloc_empty( ) {
  comp_type *node = util_malloc(sizeof *node , __func__);
  node->well      = NULL;
  return node;
}


static comp_type * comp_alloc(const char **token_list) {
  comp_type * node = comp_alloc_empty();
  comp_set_from_string(node , token_list);
  return node;
}


static void comp_free(comp_type *comp) {
  free(comp->well);
  free(comp);
}


static void comp_free__(void *__comp) {
  comp_type *comp = (comp_type *) __comp;
  comp_free(comp);
}

//
//static void comp_sched_fwrite(const comp_type *comp , int kw_size , FILE *stream) {
//  util_fwrite_string(comp->well            , stream);
//  util_fwrite_string(comp->comp_string     , stream);
//  util_fwrite_string(comp->well_dir_string , stream);
//
//  util_fwrite(&comp->i  	   , sizeof comp->i  	       	, 1 	  , stream , __func__);
//  util_fwrite(&comp->j  	   , sizeof comp->j  	       	, 1 	  , stream , __func__);
//  util_fwrite(&comp->k1 	   , sizeof comp->k1 	       	, 1 	  , stream , __func__);
//  util_fwrite(&comp->k2 	   , sizeof comp->k2 	       	, 1 	  , stream , __func__);
//  util_fwrite(&comp->sat_table     , sizeof comp->sat_table    	, 1 	  , stream , __func__);
//  util_fwrite(&comp->conn_factor   , sizeof comp->conn_factor  	, 1 	  , stream , __func__);
//  util_fwrite(&comp->well_diameter , sizeof comp->well_diameter	, 1 	  , stream , __func__);
//  util_fwrite(&comp->eff_perm      , sizeof comp->eff_perm	, 1 	  , stream , __func__);
//  util_fwrite(&comp->skin_factor   , sizeof comp->skin_factor   , 1 	  , stream , __func__);
//  util_fwrite(&comp->D_factor      , sizeof comp->D_factor	, 1 	  , stream , __func__);
//  util_fwrite(&comp->well_dir      , sizeof comp->well_dir      , 1 	  , stream , __func__);
//  util_fwrite(&comp->r0            , sizeof comp->r0            , 1 	  , stream , __func__);
//  util_fwrite(&comp->conn_factor__ , sizeof comp->conn_factor__ , 1 	  , stream , __func__);
//  util_fwrite(comp->def            , sizeof * comp->def         , kw_size , stream , __func__);
//}
//
//
//static comp_type * comp_sched_fread_alloc(int kw_size , FILE * stream) {
//  comp_type * comp = comp_alloc_empty(kw_size);
//  comp->well        	= util_fread_alloc_string( stream );
//  comp->comp_string 	= util_fread_alloc_string( stream );
//  comp->well_dir_string = util_fread_alloc_string( stream );
//
//  util_fread(&comp->i  	      	   , sizeof comp->i  	     	  , 1 	    , stream , __func__);
//  util_fread(&comp->j  	      	   , sizeof comp->j  	     	  , 1 	    , stream , __func__);
//  util_fread(&comp->k1 	      	   , sizeof comp->k1 	     	  , 1 	    , stream , __func__);
//  util_fread(&comp->k2 	      	   , sizeof comp->k2 	     	  , 1 	    , stream , __func__);
//  util_fread(&comp->sat_table      , sizeof comp->sat_table       , 1 	    , stream , __func__);
//  util_fread(&comp->conn_factor    , sizeof comp->conn_factor     , 1 	    , stream , __func__);
//  util_fread(&comp->well_diameter  , sizeof comp->well_diameter   , 1 	    , stream , __func__);
//  util_fread(&comp->eff_perm       , sizeof comp->eff_perm	  , 1 	    , stream , __func__);
//  util_fread(&comp->skin_factor    , sizeof comp->skin_factor     , 1 	    , stream , __func__);
//  util_fread(&comp->D_factor       , sizeof comp->D_factor	  , 1 	    , stream , __func__);
//  util_fread(&comp->well_dir       , sizeof comp->well_dir        , 1 	    , stream , __func__);
//  util_fread(&comp->r0             , sizeof comp->r0              , 1 	    , stream , __func__);
//  util_fread(&comp->conn_factor__  , sizeof comp->conn_factor__   , 1 	    , stream , __func__);
//  util_fread(comp->def             , sizeof * comp->def           , kw_size , stream , __func__);
//    
//  return comp;
//}

/*****************************************************************/


//void sched_kw_compdat_update_well_set(const sched_kw_compdat_type * kw , set_type * well_set) {
//  list_node_type *comp_node = list_get_head(kw->comp_list);
//  while (comp_node != NULL) {
//    comp_type * comp = list_node_value_ptr(comp_node);
//    set_add_key(well_set , comp->well);
//    comp_node = list_node_get_next(comp_node);
//  }
//}
//
//
//
//
//void sched_kw_compdat_init_conn_factor(sched_kw_compdat_type * kw , const ecl_kw_type *permx_kw, const ecl_kw_type * permz_kw , const int * dims , const int * index_field , bool *OK) {
//  float *permx = ecl_kw_get_float_ptr(permx_kw);
//  float *permz = ecl_kw_get_float_ptr(permz_kw);
//  list_node_type *comp_node = list_get_head(kw->comp_list);
//  while (comp_node != NULL) {
//    comp_type * comp = list_node_value_ptr(comp_node);
//    comp_sched_init_conn_factor(comp , permx , permz , dims , index_field , OK);
//    comp_node = list_node_get_next(comp_node);
//  }
//}
//
//
//void sched_kw_compdat_set_conn_factor(sched_kw_compdat_type * kw , const float *permx , const float *permz , const int * dims , const int * index_field) {
//  list_node_type *comp_node = list_get_head(kw->comp_list);
//  while (comp_node != NULL) {
//    comp_type * comp = list_node_value_ptr(comp_node);
//    comp_sched_set_conn_factor(comp , permx , permz , dims , index_field);
//    comp_node = list_node_get_next(comp_node);
//  }
//}



void sched_kw_compdat_fprintf(const sched_kw_compdat_type *kw , FILE *stream) {
  fprintf(stream , "COMPDAT\n");
  {
    int index;
    for (index = 0; index < vector_get_size( kw->completions ); index++) {
      const comp_type * comp = vector_iget_const( kw->completions , index );
      comp_sched_fprintf(comp , stream);
    }
  }
  fprintf(stream , "/\n\n");
}



sched_kw_compdat_type * sched_kw_compdat_alloc( ) {
  sched_kw_compdat_type * kw = util_malloc(sizeof *kw , __func__);
  kw->completions = vector_alloc_new();
  kw->__type_id   = SCHED_KW_COMPDAT_ID; 
  return kw;
}


sched_kw_compdat_type * sched_kw_compdat_safe_cast( void * arg ) {
  sched_kw_compdat_type * kw = (sched_kw_compdat_type * ) arg;
  if (kw->__type_id == SCHED_KW_COMPDAT_ID)
    return kw;
  else {
    util_abort("%s: runtime cast failed \n",__func__);
    return NULL;
  }
}



void sched_kw_compdat_add_line(sched_kw_compdat_type * kw , const char * line) {
  int tokens;
  char **token_list;
  
  sched_util_parse_line(line , &tokens , &token_list , COMPDAT_NUM_KW , NULL);
  {
    comp_type * comp = comp_alloc((const char **) token_list);
    vector_append_owned_ref(kw->completions , comp , comp_free__);
  }
  
  util_free_stringlist(token_list , tokens);
}


sched_kw_compdat_type * sched_kw_compdat_fscanf_alloc(FILE * stream, bool * at_eof, const char * kw_name) {
  bool   at_eokw = false;
  sched_kw_compdat_type * kw = sched_kw_compdat_alloc();
  
  while(!*at_eof && !at_eokw) {
    char * line = sched_util_alloc_next_entry(stream, at_eof, &at_eokw);

    if(at_eokw)
      break;
    else if (*at_eof)
      util_abort("%s: Reached EOF before COMPDAT was finished - aborting.\n", __func__);
    else 
      sched_kw_compdat_add_line(kw, line);
    
    util_safe_free(line);
  }
  return kw;
}


void sched_kw_compdat_free(sched_kw_compdat_type * kw) {
  vector_free(kw->completions);
  free(kw);
}


void sched_kw_compdat_fwrite(const sched_kw_compdat_type *kw , FILE *stream) {
  util_abort("%s: not implemented \n",__func__);
}



sched_kw_compdat_type * sched_kw_compdat_fread_alloc(FILE *stream) {
  util_abort("%s: not implemented \n",__func__);
  return NULL;
}
  


/*****************************************************************/
KW_IMPL(compdat)

