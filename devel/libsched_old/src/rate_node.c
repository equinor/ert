#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <rate_node.h>
#include <sched_util.h>
#include <string.h>
#include <util.h>
#include <ecl_sum.h>
#include <ecl_well_vars.h>


typedef enum {LRAT , ORAT , RESV , GRAT , WRAT , BHP , THP , MODE_SHUT}  well_control_type;       
typedef enum {OPEN , STOP , SHUT}  well_state_type;
static const double RATE_ERROR = -1.0;


static const int ORAT_index = 3;
static const int WRAT_index = 4;
static const int GRAT_index = 5;
static const int THP_index  = 8;
static const int BHP_index  = 9;

const char *well_control_list[] = {"LRAT" , "ORAT" , "RESV"};



struct rate_struct {
  char             *well;
  well_control_type cmode;
  well_state_type   state;   
  double 	    ORAT;         
  double 	    WRAT;    
  double 	    GRAT;    
  int               VFPTable;
  double            ALift;
  double 	    THP;	    
  double 	    BHP;	    
  double            WGASRAT;
  bool             *def;

  int               kw_size;
};


/*****************************************************************/

/*
  1: ORAT
  2: WRAT
  3: GRAT
  4: LRAT
  5: RESV
  6: THP
  7: BHP

  --------------  The modes from 11 an onwards have been dropped.
  11: GOR Penalty
  12: Drawdown
  30: Availability
  31: REIN
  32: TMRA
  33: WGRA
*/




/*
  We recognize the strings from ECLIPSE, but the mapping from ECLIPSE
  numbers to internal numbers (enum ) is arbitrary.
*/


static well_state_type __get_well_mode(const char * well_mode) {
  well_control_type cmode;
  if (strcmp(well_mode , "RESV") == 0) 
    cmode = RESV;
  else if (strcmp(well_mode , "LRAT") == 0) 
    cmode = LRAT;
  else if (strcmp(well_mode , "ORAT") == 0) 
    cmode = ORAT;
  else if (strcmp(well_mode , "WRAT") == 0)
    cmode = WRAT;
  else if (strcmp(well_mode , "GRAT") == 0)
    cmode = GRAT;
  else if (strcmp(well_mode , "THP") == 0)
    cmode = THP;
  else if (strcmp(well_mode , "BHP") == 0)
    cmode = BHP;
  /* Here is a mixing of control mode and state of a well.*/
  else if (strcmp(well_mode , "SHUT") == 0)
    cmode = MODE_SHUT;
  /*
    Some modes ignored here - check ECLIPSE documentation at page 156.
  */
  else {
    fprintf(stderr,"%s: unknown well_mode:%s - aborting \n",__func__ , well_mode);
    abort();
  }
  return cmode;
}


const char * rate_get_mode_string(const rate_type * rate) {
  const char * mode;
  switch (rate->cmode) {
  case(LRAT):
    mode = "LRAT";
    break;
  case(ORAT):
    mode = "ORAT";
    break;
  case(RESV):
    mode = "RESV";
    break;
  case(GRAT):
    mode = "GRAT";
    break;
  case(WRAT):
    mode = "WRAT";
    break;
  case(BHP):
    mode = "BHP";
    break;
  case(THP):
    mode = "THP";
    break;
  case(MODE_SHUT):
    mode = "ORAT";  /* ???? */
    break;
  default:
    fprintf(stderr,"%s well_mode:%d not recognized - aborting \n" , __func__ , rate->cmode);
    abort();
  }
  return mode;
}


static well_state_type __get_well_state_type(const char * well_state) {
  well_state_type state;
  
  if (strcmp(well_state, "OPEN") == 0) 
    state = OPEN;
  else if (strcmp(well_state, "STOP") == 0) 
    state = STOP;
  else if (strcmp(well_state, "SHUT") == 0)
    state = SHUT;
  else {
    fprintf(stderr,"%s: well_state:%s not recognized - aborting \n",__func__ , well_state);
    abort();
  }
  return state;
}


const char * rate_get_well_state_string(const rate_type * rate) {
  const char * well_state;
  switch (rate->state) {
  case(OPEN):
    well_state = "OPEN";
    break;
  case(SHUT):
    well_state = "SHUT";
    break;
  case(STOP):
    well_state = "STOP";
    break;
  default:
    fprintf(stderr,"%s well_state:%d not recognized - abortning \n" , __func__ , rate->state);
    abort();
  }
  return well_state;
}



/*****************************************************************/

static rate_type * rate_alloc_empty(int kw_size) {
  rate_type *node    = malloc(sizeof *node);
  node->well         = NULL;
  node->def          = calloc(kw_size , sizeof *node->def);
  node->kw_size      = kw_size;
  return node;
}

static double __sum_get(const ecl_sum_type * sum , int report_nr , const char * well , const char * var) {
  double value;

  if (ecl_sum_has_well_var(sum , well , var)) 
    value = ecl_sum_get_well_var(sum , report_nr , well , var);
  else {
    fprintf(stderr,"%s: warning well/variable combination:%s/%s does not exist - default value -1 used.\n",__func__ , well , var);
    value = -1;
  }
  return value;
}


rate_type * rate_alloc_from_summary(bool history_mode , const ecl_sum_type * sum , int report_nr , const char * well) {
  int  cmode = -1;

  if (ecl_sum_has_well_var(sum , well , "WMCTL")) 
    cmode = round(ecl_sum_get_well_var(sum , report_nr , well , "WMCTL"));
  else 
    util_abort("%s: summary data did not contain kewyord WMCTL - can not determine well status. \n",__func__);
  
  {
    rate_type * rate = rate_alloc_empty(11);
    rate->well = util_alloc_string_copy(well);
    if (history_mode) {
      rate->ORAT = __sum_get(sum , report_nr , well , "WOPRH");
      rate->WRAT = __sum_get(sum , report_nr , well , "WWPRH");
      rate->GRAT = __sum_get(sum , report_nr , well , "WGPRH");
      rate->BHP  = __sum_get(sum , report_nr , well , "WBHPH");
      rate->THP  = __sum_get(sum , report_nr , well , "WTHPH");
    } else {
      rate->ORAT = __sum_get(sum , report_nr , well , "WOPR" );
      rate->WRAT = __sum_get(sum , report_nr , well , "WWPR" );
      rate->GRAT = __sum_get(sum , report_nr , well , "WGPR" );
      rate->BHP  = __sum_get(sum , report_nr , well , "WBHP" );
      rate->THP  = __sum_get(sum , report_nr , well , "WTHP" );
    }

    rate->WGASRAT = -1; /* Fuck this  Kw: WWGPR */
    {
      /*
	This translates from WMCTL according to the the ECLIPSE
	documentation of the WMCTL summary variable to the internal
	representation of control mode - the two are independent.
      */
      switch (cmode) {
      case(0):
      rate->cmode = __get_well_mode("SHUT");
      rate->state = SHUT;
      break;
      case(1):
	rate->cmode = __get_well_mode("ORAT");
	rate->state = OPEN;
	break;
      case(2):
	rate->cmode = __get_well_mode("WRAT");
	rate->state = OPEN;
	break;
      case(3):
	rate->cmode = __get_well_mode("GRAT");
	rate->state = OPEN;
	break;
      case(4):
	rate->cmode = __get_well_mode("LRAT");
	rate->state = OPEN;
	break;
      case(5):
	rate->cmode = __get_well_mode("RESV");
	rate->state = OPEN;
	break;
      case(6):
	rate->cmode = __get_well_mode("THP");
	rate->state = OPEN;
	break;
      case(7):
	rate->cmode = __get_well_mode("BHP");
	rate->state = OPEN;
	break;
      default:
	fprintf(stderr,"%s: WCMTL:%d is not recognized - aborting \n",__func__ , cmode);
	abort();
      }
    }
    return rate;
  }
}
			   


const char * rate_get_well_ref(const rate_type * rate) {
  return rate->well;
}


void rate_sched_fwrite(const rate_type *rate , FILE *stream) {
  util_fwrite(&rate->kw_size	 , sizeof rate->kw_size , 1 , stream , __func__);

  util_fwrite_string(rate->well , stream);
  
  util_fwrite(&rate->cmode    , sizeof rate->cmode     , 1 	       , stream ,__func__);
  util_fwrite(&rate->state    , sizeof rate->state     , 1 	       , stream ,__func__);
  util_fwrite(&rate->ORAT     , sizeof rate->ORAT      , 1 	       , stream ,__func__);
  util_fwrite(&rate->WRAT     , sizeof rate->WRAT      , 1 	       , stream ,__func__);
  util_fwrite(&rate->GRAT     , sizeof rate->GRAT      , 1 	       , stream ,__func__);
  util_fwrite(&rate->VFPTable , sizeof rate->VFPTable  , 1 	       , stream ,__func__);
  util_fwrite(&rate->ALift    , sizeof rate->ALift     , 1 	       , stream ,__func__);
  util_fwrite(&rate->THP      , sizeof rate->THP       , 1 	       , stream ,__func__);
  util_fwrite(&rate->BHP      , sizeof rate->BHP       , 1 	       , stream ,__func__);
  util_fwrite(&rate->WGASRAT  , sizeof rate->WGASRAT   , 1 	       , stream ,__func__);  
  util_fwrite(rate->def       , sizeof * rate->def     , rate->kw_size , stream ,__func__);
  
}




rate_type * rate_sched_fread_alloc(FILE *stream) {
  rate_type *rate;
  int kw_size;

  util_fread(&kw_size  , sizeof kw_size , 1 , stream , __func__);
  rate  = rate_alloc_empty(kw_size);
  rate->well         = util_fread_alloc_string( stream ); 
  
  util_fread(&rate->cmode     , sizeof rate->cmode        , 1 		  , stream , __func__);
  util_fread(&rate->state     , sizeof rate->state 	  , 1 		  , stream , __func__);
  util_fread(&rate->ORAT      , sizeof rate->ORAT  	  , 1 		  , stream , __func__);
  util_fread(&rate->WRAT      , sizeof rate->WRAT  	  , 1 		  , stream , __func__);
  util_fread(&rate->GRAT      , sizeof rate->GRAT  	  , 1 		  , stream , __func__);
  util_fread(&rate->VFPTable  , sizeof rate->VFPTable     , 1 		  , stream , __func__);
  util_fread(&rate->ALift     , sizeof rate->ALift        , 1 		  , stream , __func__);
  util_fread(&rate->THP	      , sizeof rate->THP 	  , 1 		  , stream , __func__);
  util_fread(&rate->BHP	      , sizeof rate->BHP 	  , 1 		  , stream , __func__);
  util_fread(&rate->WGASRAT   , sizeof rate->WGASRAT      , 1 		  , stream , __func__);  
  util_fread(rate->def        , sizeof * rate->def        , rate->kw_size , stream , __func__);
  
  return rate;
}




void rate_sched_fprintf(const rate_type * rate , FILE *stream) {
  fprintf(stream , "  ");
  sched_util_fprintf_qst(rate->def[0]  , rate->well          , 8 , stream);
  sched_util_fprintf_qst(rate->def[1]  , rate_get_well_state_string(rate) , 4 , stream);
  sched_util_fprintf_qst(rate->def[2]  , rate_get_mode_string(rate)       , 4 , stream);
  sched_util_fprintf_dbl(rate->def[3]  , rate->ORAT     , 12 , 3 , stream);
  sched_util_fprintf_dbl(rate->def[4]  , rate->WRAT     , 12 , 3 , stream);
  sched_util_fprintf_dbl(rate->def[5]  , rate->GRAT     , 12 , 3 , stream);
  sched_util_fprintf_int(rate->def[6]  , rate->VFPTable , 4      , stream);
  sched_util_fprintf_dbl(rate->def[7]  , rate->ALift    , 12 , 3 , stream);
  sched_util_fprintf_dbl(rate->def[8]  , rate->THP      , 10 , 4 , stream);
  sched_util_fprintf_dbl(rate->def[9]  , rate->BHP      , 10 , 4 , stream);
  sched_util_fprintf_dbl(rate->def[10] , rate->WGASRAT  , 10 , 4 , stream);
  fprintf(stream , " /\n");
}


void rate_sched_fprintf_rates(const rate_type * rate , FILE *stream) {
  fprintf(stream , "  %8s %15.3f %15.3f %15.3f ",rate->well , rate->ORAT , rate->WRAT , rate->GRAT);
  if (rate->def[THP_index])
    fprintf(stream , "%15.3f ",RATE_ERROR);
  else
    fprintf(stream , "%15.3f ",rate->THP);
  if (rate->def[BHP_index])
    fprintf(stream , "%15.3f ",RATE_ERROR);
  else
    fprintf(stream , "%15.3f ",rate->BHP);
  fprintf(stream,"\n");
}


void rate_fprintf(const rate_type * rate , FILE * stream) {
  fprintf(stream,"%s  ORAT:%g  \n",rate->well , rate->ORAT);
}


static void rate_set_from_string(rate_type * node , int kw_size , const char **token_list) {
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
  
  node->ORAT     = sched_util_atof( token_list[3] );
  node->WRAT     = sched_util_atof( token_list[4] );
  node->GRAT     = sched_util_atof( token_list[5] );
  node->VFPTable = sched_util_atoi( token_list[6] );
  node->ALift    = sched_util_atof( token_list[7] );
  node->THP      = sched_util_atof( token_list[8] );
  node->BHP      = sched_util_atof( token_list[9] );
  node->WGASRAT  = sched_util_atof( token_list[10]);

  node->state = __get_well_state_type( token_list[1] );
  node->cmode = __get_well_mode( token_list[2] );
}



rate_type * rate_alloc_from_token_list(int kw_size , const char **token_list) {
  rate_type *node = rate_alloc_empty(kw_size);
  rate_set_from_string(node ,kw_size , token_list);
  return node;
}


void rate_free(rate_type *rate) {
  free(rate->well);
  free(rate->def);
  free(rate);
}


void rate_free__(void *__rate) {
  rate_type *rate = (rate_type *) __rate;
  rate_free(rate);
}

rate_type * rate_copyc(const rate_type * src) {
  rate_type * new = rate_alloc_empty(src->kw_size);
  new->well = util_alloc_string_copy(src->well);
  new->cmode = src->cmode;
  new->state = src->state;
  new->ORAT = src->ORAT;         
  new->WRAT = src->WRAT;    
  new->GRAT = src->GRAT;    
  new->VFPTable = src->VFPTable;
  new->ALift = src->ALift;
  new->THP = src->THP;	    
  new->BHP = src->BHP;	    
  new->WGASRAT = src->WGASRAT;
  memcpy(new->def , src->def , new->kw_size * sizeof * new->def);
  return new;
}

const void * rate_copyc__(const void *void_rate) {
  return rate_copyc((const rate_type *) void_rate);
}

/*****************************************************************/

const char * rate_node_get_well_ref(const rate_type * rate) { return rate->well; }

double rate_get_ORAT(const rate_type * rate, bool *def) {
  return rate->ORAT;
}


double rate_get_GRAT(const rate_type * rate, bool *def) {
  return rate->GRAT;
}


double rate_get_WRAT(const rate_type * rate, bool *def) {
  return rate->WRAT;
}


double rate_get_BHP(const rate_type * rate, bool *def) {

  if (rate->def[BHP_index])
    *def = true;
  
  return rate->BHP;
}


double rate_get_THP(const rate_type * rate, bool *def) {

  if (rate->def[THP_index])
    *def = true;
  
  return rate->THP;
}


double rate_get_GOR(const rate_type * rate, bool *error, bool *def) {
  double GOR;
  if (rate->ORAT != 0.0) {
    GOR = rate->GRAT / rate->ORAT;
    *error = false;
  } else {
    if (rate->GRAT == 0.0) {
      GOR = 0.0;
      *error = false;
    } else {
      GOR = RATE_ERROR;
      *error = true;
    }
  }
  return GOR;
}


double rate_get_WCT(const rate_type * rate, bool *error, bool *def) {
  double WCT;
  if ((rate->ORAT + rate->WRAT) != 0.0) {
    WCT = rate->WRAT / (rate->ORAT + rate->WRAT);
    *error = false;
  } else {
    WCT = 0.0;
    *error = false;
  }
  return WCT;
}




double rate_iget(const rate_type * rate , well_var_type var_type , bool *error , bool *def) {
  switch (var_type) {
  case(well_var_orat):
    return rate_get_ORAT(rate , def);
    break;
  case(well_var_grat):
    return rate_get_GRAT(rate , def);
    break; 
  case(well_var_wrat):
    return rate_get_WRAT(rate , def);
    break;
  case(well_var_gor):
    return rate_get_GOR(rate ,  error , def);
    break;
  case(well_var_wct):
    return rate_get_WCT(rate , error , def);
    break;
  case(well_var_bhp):
    return rate_get_BHP(rate , def);
    break;
  case(well_var_thp):
    return rate_get_THP(rate , def);
    break;
  default:
    fprintf(stderr,"%s: Internal error var_type = %d is not recognized - aborting \n",__func__ , var_type);
    abort();
  }
}
