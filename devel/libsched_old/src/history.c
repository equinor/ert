/*
  Denne koden har et helvetes kaos med om date_nr skal starte på 1 eller 0.
*/

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <hash.h>
#include <sched_util.h>
#include <history.h>
#include <util.h>
#include <sched_file.h>
#include <sched_kw.h>
#include <ecl_sum.h>
#include <rate_node.h>
#include <date_node.h>

typedef struct history_node_struct history_node_type;

struct history_struct {
  int  		      size;
  int  		      alloc_size; 
  time_t              start_date;
  time_t              creation_time;
  char               *data_src;
  bool                history_mode;
  history_node_type **data;
  hash_type          *well_hash;
};



struct history_node_struct {
  hash_type       * data;
  date_node_type  * date;
};



/* 
   This code has lots of time_step - 1 to meet ECLIPSE - UGGLY.
*/


/*****************************************************************/

history_node_type * history_node_alloc() {
  history_node_type * history_node = malloc(sizeof * history_node);
  history_node->data = hash_alloc();
  return history_node;
}


void history_node_free(history_node_type * history_node) {
  hash_free(history_node->data);
  date_node_free(history_node->date);
  free(history_node);
}


void history_node_fwrite(const history_node_type * history_node , FILE *stream) {
  bool isNULL;
  if (history_node == NULL)
    isNULL = true;
  else
    isNULL = false;
  
  util_fwrite(&isNULL , sizeof isNULL , 1 , stream , __func__);
  if (!isNULL) {
    char **well_list;
    int    wells , i ;
    date_node_fwrite(history_node->date , stream);

    wells = hash_get_size(history_node->data);
    well_list = hash_alloc_keylist(history_node->data);
    util_fwrite(&wells , sizeof wells , 1 , stream , __func__);
    for (i = 0; i < wells; i++)
      rate_sched_fwrite(hash_get(history_node->data , well_list[i]) , stream);

    util_free_stringlist(well_list , hash_get_size(history_node->data));
  }
}
	

history_node_type * history_node_fread_alloc(const time_t * start_date , FILE *stream) {
  history_node_type *history_node = NULL;
  bool isNULL;
  util_fread(&isNULL , sizeof isNULL , 1 , stream , __func__);
  if (!isNULL) {
    int wells , i;
    rate_type *rate;
    bool stop;
    history_node = history_node_alloc();
    history_node->date = date_node_fread_alloc(start_date , -1 , -1 , stream , &stop);
    util_fread(&wells , sizeof wells , 1 , stream , __func__);
    for (i=0; i < wells; i++) {
      rate = rate_sched_fread_alloc(stream);
      hash_insert_hash_owned_ref(history_node->data , rate_get_well_ref(rate) , rate , rate_free__);
    }
  }
  return history_node;
}
							 

void history_node_fprintf(const history_node_type * hist_node, FILE * stream) {
  date_node_fprintf(hist_node->date , stream , -1 , -1 , NULL);
  {
    char **key_list  = hash_alloc_keylist(hist_node->data);
    rate_type * rate;
    int i;
    for (i = 0; i < hash_get_size(hist_node->data); i++) {
      rate = hash_get(hist_node->data , key_list[i]);
      rate_fprintf(rate , stream);
    }
    util_free_stringlist(key_list , hash_get_size(hist_node->data));
  }
}



/*****************************************************************/

/*
  Some pain, to ensure that all newly (re) allocated pointers 
  are initialised with NULL.
*/
static void history_realloc_data(history_type * hist , int alloc_size) {
  const int new_size = alloc_size;
  const int old_size = hist->alloc_size;
  int i;
  history_node_type **new;

  new = malloc(new_size * sizeof *new);
  for (i=0; i < new_size; i++)
    new[i] = NULL;

  if (hist->data != NULL) {
    history_node_type **old = hist->data;
    memcpy(new , old , old_size * sizeof *old);
    free(old);
  }
  
  hist->data       = new;
  hist->alloc_size = new_size;

}






history_type * history_alloc(time_t start_date) {
  history_type * hist = malloc(sizeof *hist);
  hist->size       = 0;
  hist->well_hash  = hash_alloc();
  hist->data       = NULL;
  hist->alloc_size = 0;
  hist->start_date = start_date;
  hist->creation_time = time(NULL);
  hist->data_src   = NULL;
  history_realloc_data(hist , 5);
  return hist;
}



void history_fwrite(const history_type * hist , FILE *stream) {
  util_fwrite(&hist->creation_time , sizeof hist->creation_time , 1 , stream , __func__);
  util_fwrite(&hist->history_mode  , sizeof hist->history_mode  , 1 , stream , __func__);
  util_fwrite_string(hist->data_src                        , stream);
  util_fwrite(&hist->size          , sizeof hist->size          , 1 , stream , __func__);
  util_fwrite(&hist->alloc_size    , sizeof hist->alloc_size    , 1 , stream , __func__);
  util_fwrite(&hist->start_date    , sizeof hist->start_date    , 1 , stream , __func__);
  {
    int i , wells;
    char **well_list;
    for (i=0; i < hist->size; i++)
      history_node_fwrite(hist->data[i] , stream);

    wells = hash_get_size(hist->well_hash);
    util_fwrite(&wells , sizeof wells , 1 , stream , __func__);
    well_list = hash_alloc_keylist(hist->well_hash);
    for (i=0; i < wells; i++) 
      util_fwrite_string(well_list[i] , stream);
    
    util_free_stringlist(well_list , hash_get_size(hist->well_hash));
  }
}


history_type * history_fread_alloc(FILE *stream) {
  history_type * hist = history_alloc(0);
  
  util_fread(&hist->creation_time , sizeof hist->creation_time , 1 , stream , __func__);
  util_fread(&hist->history_mode  , sizeof hist->history_mode  , 1 , stream , __func__);
  hist->data_src = util_fread_alloc_string(stream);
  util_fread(&hist->size          , sizeof hist->size       , 1 , stream , __func__);
  util_fread(&hist->alloc_size    , sizeof hist->alloc_size , 1 , stream , __func__);
  util_fread(&hist->start_date    , sizeof hist->start_date , 1 , stream , __func__);
  history_realloc_data(hist , hist->alloc_size);
  {
    int i , wells;
    char *well;
    for (i=0; i < hist->size; i++)
      hist->data[i] = history_node_fread_alloc(&hist->start_date , stream);
    
    util_fread(&wells , sizeof wells , 1 , stream , __func__);
    for (i=0; i < wells; i++) {
      well = util_fread_alloc_string(stream);
      hash_insert_int(hist->well_hash , well , 1);
      free(well);
    }
  }
  return hist;
}


history_node_type * history_get_node(const history_type * hist , int time_step) {
  return hist->data[time_step - 1];
}


history_node_type * history_add_node(history_type * hist , int time_step) {
  hist->data[time_step-1] = history_node_alloc();
  return hist->data[time_step - 1];
}


bool history_has_well(const history_type * hist , const char * well) {
  return hash_has_key(hist->well_hash , well);
}
    

void history_summarize(const history_type * hist , FILE * stream) {
  char ** well_list = hash_alloc_keylist(hist->well_hash);
  int iw , itime;
  printf("-----------------------------------------------------------------\n");
  for (iw = 0; iw < hash_get_size(hist->well_hash); iw++) 
    printf("Well[%d] : <%s> \n",iw , well_list[iw]);
  printf("-----------------------------------------------------------------\n");

  for (itime = 0; itime < hist->size; itime++) {
    const history_node_type * node = hist->data[itime];
    printf("History node: %d ",itime);
    history_node_fprintf(node , stream);
  }
  util_free_stringlist(well_list , hash_get_size( hist->well_hash ));

}


/**
This function expects time_step in the interval [1,hist->size]
*/
static const rate_type * history_get_rate_node(const history_type * hist , int time_step, const char * well) {
  const rate_type * rate = NULL;

  if (time_step <= hist->size) {
    const history_node_type * history_node = history_get_node(hist , time_step); 
    if (history_node != NULL) {
      if (hash_has_key(history_node->data , well)) 
	rate = hash_get(history_node->data , well);
      else {
	if (!history_has_well(hist , well)) {
          history_summarize(hist , stdout);
	  util_abort("%s: The well:%s does not exist in the history object - aborting \n",__func__ , well);
	}
      }
    } else 
      util_abort("%s: asked for timestep:%d ... \n",__func__ , time_step);
  } else 
    util_abort("%s tried to ask for nonexistning time_step:%d - aborting \n",__func__ , time_step);
  
  return rate;
}


static history_node_type * history_get_new_node(history_type * hist , int date_nr) {
  history_node_type * history_node;

  if (date_nr > hist->alloc_size) 
    history_realloc_data(hist , (hist->alloc_size + date_nr));
  
  history_node = history_get_node(hist , date_nr);
  if (history_node == NULL)
    history_node = history_add_node(hist , date_nr);

  if (date_nr > hist->size)
    hist->size = date_nr;

  return history_node;
}


void history_add_date(history_type * hist, const date_node_type * date) {
  int date_nr = date_node_get_date_nr(date);
  history_node_type * history_node = history_get_new_node(hist , date_nr);
  history_node->date = date_node_copyc(date);
}


void history_add_rate(history_type * hist , int date_nr , const rate_type * rate) {
  history_node_type * history_node = history_get_new_node(hist , date_nr);
  if (hash_has_key(history_node->data , rate_node_get_well_ref(rate))) {
    fprintf(stderr,"%s: Warning: tried to add well rate for:%s twice - only last instance stored\n",__func__ , rate_node_get_well_ref(rate));
    /*
      fprintf(stderr,"%s: INTERNAL error - tried adding the same well:%s rate twice - aborting \n",__func__ , );
      abort();
    */
    hash_del(history_node->data , rate_node_get_well_ref(rate));
  }
  
  hash_insert_copy(history_node->data , rate_node_get_well_ref(rate) , rate , rate_copyc__ , rate_free__);
  {
    const char * well = rate_get_well_ref(rate);
    if (!hash_has_key(hist->well_hash , well))
      hash_insert_int(hist->well_hash , well , 1);
  }

}


/*****************************************************************/


double history_get_ORAT(const history_type * hist , int time_step , const char * well, bool *def) {
  const rate_type *rate = history_get_rate_node(hist , time_step , well);
  if (rate == NULL)
    return 0.0;
  else 
    return rate_get_ORAT(rate , def);
}

double history_get_GRAT(const history_type * hist , int time_step , const char * well, bool *def) {
  const rate_type *rate = history_get_rate_node(hist , time_step , well);
  if (rate == NULL)
    return 0.0;
  else 
    return rate_get_GRAT(rate , def);
}

double history_get_WRAT(const history_type * hist , int time_step , const char * well, bool *def) {
  const rate_type *rate = history_get_rate_node(hist , time_step , well);
  if (rate == NULL)
    return 0.0;
  else 
    return rate_get_WRAT(rate , def);
}


double history_get_GOR(const history_type * hist , int time_step , const char * well , bool *error, bool *def) {
  const rate_type *rate = history_get_rate_node(hist , time_step , well);
  if (rate == NULL)
    return 0.0;
  else 
    return rate_get_GOR(rate , error , def);
}


double history_get_WCT(const history_type * hist , int time_step , const char * well , bool *error, bool *def) {
  const rate_type *rate = history_get_rate_node(hist , time_step , well);
  if (rate == NULL)
    return 0.0;
  else 
    return rate_get_WCT(rate , error , def);
}


double history_iget(const history_type * hist , int time_step , const char * well , int var_index , bool *error , bool *def) {
  const rate_type * rate = history_get_rate_node(hist , time_step , well);
  if (rate == NULL)
    return 0.0;
  else 
    return rate_iget(rate , var_index , error , def);
}



static well_var_type history_get_var_type(const history_type * hist , const char * var) {
  well_var_type var_type;
  if (!ecl_well_var_valid(var , &var_type)) {
    util_abort("%s: variable: %s not recognized\n",__func__ , var);
    return 0;  /* To keep the compiler happy */
  } else
    return var_type;
}



double history_get(const history_type * hist , int report_step , const char * well , const char * var) {
  bool error;
  bool def;
  
  return history_iget(hist , report_step , well , history_get_var_type(hist , var) , &error , &def);
}


double history_get2(const history_type * hist , int report_step , const char * well , const char * var , bool *default_used) {
  bool error;
  *default_used = false;
  return history_iget(hist , report_step , well , history_get_var_type(hist , var) , &error , default_used);
}
  


char ** history_alloc_well_list(const history_type *hist,  int report_step) {
  int Nwells              = hash_get_size(hist->well_hash);
  char ** total_well_list = hash_alloc_keylist(hist->well_hash);
  char ** well_list;
  int active = 0;
  int iw;
  for (iw = 0; iw < Nwells; iw++) {
    if (history_get_rate_node(hist , report_step , total_well_list[iw])) 
      active++;
  }
  well_list = malloc(active * sizeof * well_list);

  active = 0;
  for (iw = 0; iw < Nwells; iw++) {
    if (history_get_rate_node(hist , report_step , total_well_list[iw])) {
      well_list[active] = util_alloc_string_copy(total_well_list[iw]);
      active++;
    }
  }
  util_free_stringlist( total_well_list , hash_get_size( hist->well_hash ));
  return well_list;
}


history_type * history_alloc_from_schedule(const sched_file_type *s) {
  history_type *hist              = history_alloc(sched_file_get_start_date(s));
  list_node_type *list_node    = list_get_head(sched_file_get_kw_list(s));
  date_node_type *current_date = NULL;
  
  while (list_node != NULL) {
    const sched_kw_type * sched_kw = list_node_value_ptr(list_node);
    sched_kw_make_history(sched_kw , hist , &current_date );
    list_node = list_node_get_next(list_node);
  }

  hist->history_mode = true;
  return hist;
}



static  history_type * history_alloc_from_summary__(const ecl_sum_type * sum , int Nwells , const char ** well_list , bool history_mode) {
  history_type *hist              = history_alloc(ecl_sum_get_start_time(sum));
  int    first_report , last_report , report_nr;
  time_t start_time = ecl_sum_get_start_time(sum);
  ecl_sum_get_report_size(sum , &first_report , &last_report);
  
  if (first_report == 0)
    first_report = 1;
  /*
    The first summary report is from time = 0 - before the
    simulation has actually started - bumping it up ...
  */
  
  if (first_report > 1) {
    fprintf(stderr,"%s: warning summary object missing the first:%d report steps - empty records prepended. \n",__func__ , first_report);
    for (report_nr = 0; report_nr < first_report; report_nr++) {
      /*
	???
      */
    }
  }
  
  
  for (report_nr = first_report; report_nr <= last_report; report_nr++) {
    date_node_type * date_node = date_node_alloc_ext(false , ecl_sum_get_sim_time(sum , report_nr) , report_nr  , &start_time);
    int iwell;
    history_add_date(hist , date_node);
    for (iwell = 0; iwell < Nwells; iwell++) {
      rate_type * rate = rate_alloc_from_summary(history_mode , sum , report_nr , well_list[iwell]);
      if (rate != NULL)
	history_add_rate(hist , report_nr , rate);
      }
  }
  hist->history_mode = history_mode;
  return hist;
}


history_type * history_alloc_from_summary(const ecl_sum_type * ecl_sum , bool history_mode) {
  history_type * history = history_alloc_from_summary__(ecl_sum , ecl_sum_get_Nwells(ecl_sum) , ecl_sum_get_well_names_ref(ecl_sum) , history_mode);
  return history;
}


time_t history_get_report_date(history_type * hist , int time_step) {
  const history_node_type * node = history_get_node(hist , time_step);
  if (node == NULL) {
    util_abort("%s: could not lookup date_node:%d - aborting \n",__func__ , time_step);
    return -1; /* To keep the compiler happy */
  } else 
    return date_node_get_date(node->date);
}


int history_get_num_reports(const history_type * hist) { return hist->size; }


void history_free(history_type *hist) {
  int i;
  for (i=0; i < hist->size; i++) {
    if (hist->data[i] != NULL)
      history_node_free(hist->data[i]);
  }
  
  if (hist->data_src != NULL)
    free(hist->data_src);
  
  free(hist->data);
  hash_free(hist->well_hash);
  free(hist);
}



