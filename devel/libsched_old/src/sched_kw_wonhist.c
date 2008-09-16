#include <stdlib.h>
#include <stdio.h>
#include <sched_kw_wconhist.h>
#include <list.h>

typedef enum {OPEN , STOP , SHUT}           well_state_type;


struct wconhist_kw_struct {
  list_type *rate_list;
};

typedef struct  {
  char            *well;
  /*char            *control_mode;*/
  well_state_type  state;   
  double 	   ORAT;    
  double 	   WRAT;    
  double 	   GRAT;    
  double 	   THP;	    
  double 	   BHP;	    
  double 	   GOR;	    
  double 	   WCT;     
} rate_type ;


/*****************************************************************/

static void rate_set1(double *rate , const char * token , double missing_value) {
  if (token == NULL)
    *rate = missing_value;
  else
    *rate = atof(token);
}


static void rate_set(rate_type * node , double missing_value , int tokens , const char **token_list , bool *well_shut) {
  node->well = util_realloc_string_copy(node->well , token_list[0]);
  node->well = dequote_string(node->well);

  rate_set1(&node->ORAT  , token_list[3],missing_value);
  rate_set1(&node->WRAT  , token_list[4],missing_value);
  rate_set1(&node->GRAT  , token_list[5],missing_value);
  rate_set1(&node->THP   , token_list[8],missing_value);
  rate_set1(&node->BHP   , token_list[9],missing_value);
  

  if (node->ORAT != 0.0)
    node->GOR = node->GRAT / node->ORAT;
  else
    node->GOR = RATE_ERROR;

  if ((node->ORAT + node->WRAT) != 0.0)
    node->WCT = node->WRAT / (node->ORAT + node->WRAT);
  else
    node->WCT = RATE_ERROR;
  
  if (strcmp(token_list[1] , "OPEN") == 0) node->state = OPEN;
  if (strcmp(token_list[1] , "STOP") == 0) node->state = STOP;
  if (strcmp(token_list[1] , "SHUT") == 0) node->state = SHUT;
}


static rate_type * rate_alloc(double missing_value , int tokens, const char **token_list) {
  rate_type *node = malloc(sizeof *node);
  bool well_shut;
  node->well = NULL;
  rate_set(node , missing_value , tokens , token_list , &well_shut);
  return node;
}


static void rate_free(rate_type *rate) {
  free(rate->well);
  free(rate);
}


static void rate_free__(void *__rate) {
  rate_type *rate = (rate_type *) __rate;
  rate_free(rate);
}



/*****************************************************************/




sched_kw_wconhist_type * sched_kw_wconhist_alloc( ) {
  sched_kw_wconhist_type * kw = malloc(sizeof *kw);
  kw->rate_list = list_alloc();
  return kw;
}



void sched_kw_wconhist_free(sched_kw_wconhist_type * kw) {
  list_free(kw);
}






