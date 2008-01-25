#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <thist.h>
#include <util.h>
#include <time.h>


typedef struct thist_node_struct thist_node_type;


struct thist_node_struct {
  thist_data_type   data_type;
  bool     	    use_true_time;
  int      	    ens_size;
  int      	    time_step;
  time_t   	    true_time;

  /* 
     This mask is not very intuitive - should probably just work with
     two bool arrays.
  */
  int     	  * set_mask;
  double  	  * forecast_data;
  double  	  * analyzed_data;

  int                forecast_set_count;
  int                analyzed_set_count;
  bool              _forecast;
  bool              _analyzed;
};


static void thist_node_clear(thist_node_type * node) {
  int i;
  for (i=0; i < node->ens_size; i++)
    node->set_mask[i] = 0;
  node->forecast_set_count = 0;
  node->analyzed_set_count = 0;
}


static void thist_node_realloc_data(thist_node_type * node, int ens_size) {
  node->ens_size      = ens_size;
  if (node->_forecast) node->forecast_data = malloc(ens_size * sizeof * node->forecast_data);
  if (node->_analyzed) node->analyzed_data = malloc(ens_size * sizeof * node->analyzed_data);
}
  

static thist_node_type * thist_node_alloc(int ens_size , int time_step , thist_data_type data_type) {
  thist_node_type * node = malloc(sizeof * node);
  node->time_step     = time_step;
  node->use_true_time = false;
  node->data_type     = data_type;
  node->forecast_data = NULL;
  node->analyzed_data = NULL;
  node->set_mask      = malloc(ens_size * sizeof * node->set_mask);
  node->_forecast     = false;
  node->_analyzed     = false;
  
  if (node->data_type & thist_forecast) node->_forecast = true; 
  if (node->data_type & thist_analyzed) node->_analyzed = true; 
  node->forecast_data = NULL;
  node->analyzed_data = NULL;
  thist_node_realloc_data(node , ens_size);
  thist_node_clear(node);
  
  return node;
}



static void thist_node_set_mask(thist_node_type * node , int iens , thist_data_type mask) {
  if ((node->set_mask[iens] & mask) == 0) { 
    node->set_mask[iens] += mask;

    if (mask == thist_forecast)
      node->forecast_set_count++;

    if (mask == thist_analyzed)
      node->analyzed_set_count++;

  }
}



static void thist_node_update_scalar(thist_node_type * node , int iens, const double * forecast , const double * analyzed) {
  if (iens >= 0 && iens < node->ens_size) {

    if (node->_forecast && forecast != NULL) {
      node->forecast_data[iens] = *forecast;
      thist_node_set_mask(node , iens , thist_forecast);
    }

    if (node->_analyzed && analyzed != NULL) {
      node->analyzed_data[iens] = *analyzed;
      thist_node_set_mask(node , iens , thist_analyzed);
    }
    
  } else {
    fprintf(stderr,"%s: tried to update ens_member:%d ens_size:%d - aborting \n",__func__ , iens , node->ens_size);
    abort();
  }
}


static void thist_node_update_vector(thist_node_type * node , const double * forecast , const double * analyzed) {
  int iens;
  if (node->_forecast && forecast != NULL) {
    memcpy(node->forecast_data , forecast , node->ens_size * sizeof * forecast);
    for (iens = 0; iens < node->ens_size; iens++)
      thist_node_set_mask(node , iens , thist_forecast);
  }

  if (node->_analyzed && analyzed != NULL) {
    memcpy(node->analyzed_data , analyzed , node->ens_size * sizeof * analyzed);
    for (iens = 0; iens < node->ens_size; iens++)
      thist_node_set_mask(node , iens , thist_analyzed);
  }
}



static void thist_node_free(thist_node_type * node) {
  if (node->_forecast) free(node->forecast_data);
  if (node->_analyzed) free(node->analyzed_data);
  free(node->set_mask);
}


static void thist_node_set_true_time(thist_node_type * node , int day, int month , int year) {
  node->true_time = util_make_time1(day , month , year);
  node->use_true_time = true;
}


static void thist_node_fwrite_data(const thist_node_type * node , thist_data_type data_type , FILE * stream) {
  int                set_size;
  double           * data;

  if (data_type == thist_forecast) {
    set_size = node->forecast_set_count;
    data     = node->forecast_data;
  } else if (data_type == thist_analyzed) {
    set_size = node->analyzed_set_count;
    data     = node->analyzed_data;
  } else {
    fprintf(stderr,"%s: internal error - aborting \n",__func__);
    abort();
  }

  util_fwrite(&set_size , sizeof set_size              , 1 , stream , __func__);
  if (set_size == node->ens_size) 
    util_fwrite(data , sizeof * data , node->ens_size , stream , __func__);
  else {
    int i;
    for (i=0; i < node->ens_size; i++) {
      if (node->set_mask[i] & data_type) 
	util_fwrite(&data[i] , sizeof data[i] , 1 , stream, __func__);
    }
  }
}


static void thist_node_matlab_dump(const thist_node_type * node , FILE *stream) {
  const int one  = 1;
  const int zero = 0;
  util_fwrite(&one                 , sizeof one                 , 1 , stream , __func__);
  util_fwrite(&node->time_step     , sizeof node->time_step     , 1 , stream , __func__);
  util_fwrite(&node->true_time     , sizeof node->true_time     , 1 , stream , __func__);
  if (node->use_true_time)
    util_fwrite(&one , sizeof one, 1 , stream , __func__);
  else
    util_fwrite(&zero , sizeof zero, 1 , stream , __func__);

  thist_node_fwrite_data(node , thist_forecast , stream);
  thist_node_fwrite_data(node , thist_analyzed , stream);
}


/*****************************************************************/



struct thist_struct {
  bool forecast;
  bool analyzed;
  thist_data_type data_type;
  int data_size;
  int ens_size;
  thist_node_type ** data;
  char             * title;
};



static void thist_realloc_data(thist_type * thist , int new_size) {
  thist_node_type **old_data;
  int               old_size = thist->data_size;
  int               i;

  old_data    = thist->data;
  thist->data = malloc(new_size * sizeof * thist->data);
 for (i=0; i < new_size; i++)
    thist->data[i] = NULL;
  
  for (i=0; i < old_size; i++)
    thist->data[i] = old_data[i];
  
  thist->data_size = new_size;
  free(old_data);
}


void thist_set_title(thist_type * thist , const char * title) {
  thist->title = util_realloc_string_copy(thist->title , title);
}
	
							   

thist_type * thist_alloc(int length , int ens_size , thist_data_type data_type , const char * title) {
  thist_type * thist = malloc(sizeof * thist);
  thist->ens_size    = ens_size;
  thist->data_type   = data_type;
  thist->data_size   = 0;
  thist->data        = NULL;
  thist->title       = NULL;
  thist_set_title(thist , title);
  thist_realloc_data(thist , length);
  return thist;
}



void thist_free(thist_type * thist) {
  int i;
  for (i=0; i < thist->data_size; i++) 
    if (thist->data[i] != NULL) thist_node_free(thist->data[i]);
  
  if (thist->title != NULL)
    free(thist->title);
  free(thist->data);
  free(thist);
}


static void thist_assert_time_step(thist_type * thist, int time_step) {
  if (time_step < 0) {
    fprintf(stderr,"%s time_step=%d < 0 INVALID \n",__func__ , time_step);
    abort();
  }

  if (time_step >= thist->data_size) 
    thist_realloc_data(thist , 2 * (time_step + 1));

  if (thist->data[time_step] == NULL)
    thist->data[time_step] = thist_node_alloc(thist->ens_size , time_step , thist->data_type);
}


static void thist_assert_forecast_analyzed__(const thist_type * thist , thist_data_type data_type) {
  if (thist->data_type != thist_both) {
    if (thist->data_type != data_type) {
      fprintf(stderr,"%s: thist->data_type:%d   input_type:%d   ---  invalid update - aborting \n",__func__ , thist->data_type , data_type);
      abort();
    }
  }
}




static void thist_update_vector__(thist_type * thist , int time_step , const double * forecast_data , const double * analyzed_data) {
  thist_assert_time_step(thist , time_step);
  thist_node_update_vector(thist->data[time_step] , forecast_data , analyzed_data);
}


static void thist_update_scalar__(thist_type * thist , int time_step , int iens , double forecast_value , double analyzed_value) {
  thist_assert_time_step(thist , time_step);
  thist_node_update_scalar(thist->data[time_step] , iens , (const double *) &forecast_value , (const double *) &analyzed_value);
}


/*
  Here comes the front ends ...
*/

void thist_update_scalar(thist_type * thist , int time_step , int iens , double forecast_value , double analyzed_value) {
  thist_assert_forecast_analyzed__(thist , thist_both);
  thist_update_scalar__(thist , time_step , iens , forecast_value , analyzed_value);
}

void thist_update_scalar_forecast(thist_type * thist , int time_step , int iens , double forecast_value) {
  thist_assert_forecast_analyzed__(thist , thist_forecast);
  thist_update_scalar__(thist , time_step , iens , forecast_value , 0);
}

void thist_update_scalar_analyzed(thist_type * thist , int time_step , int iens , double analyzed_value) {
  thist_assert_forecast_analyzed__(thist , thist_analyzed);
  thist_update_scalar__(thist , time_step , iens , 0 , analyzed_value);
}

void thist_update_vector(thist_type * thist , int time_step , const double *  forecast_data , const double *  analyzed_data) {
  thist_assert_forecast_analyzed__(thist , thist_both);
  thist_update_vector__(thist , time_step , forecast_data , analyzed_data);
}

void thist_update_vector_forecast(thist_type * thist , int time_step , const double *  forecast_data) {
  thist_assert_forecast_analyzed__(thist , thist_forecast);
  thist_update_vector__(thist , time_step , forecast_data , NULL);
}

void thist_update_vector_analyzed(thist_type * thist , int time_step , const double *  analyzed_data) {
  thist_assert_forecast_analyzed__(thist , thist_analyzed);
  thist_update_vector__(thist , time_step , NULL , analyzed_data);
}





/*
string : title
integer: length
---- * length ----
int active
int time_step
int true_time
int use_true_time
int forecast_size
double forecast_data
int analyzed_size
double analyzed_data
----
*/



/*
  matlob code to load this in ../../matlab/src/ins_plot.m
*/
void thist_matlab_dump(const thist_type * thist , const char * filename) {
  const char * title;
  char *path;
  util_alloc_file_components(filename , &path , NULL , NULL);
  if (!util_path_exists(path)) {
    printf("Making directory %s ...\n",path);
    util_make_path(path);
  }
  
  if (thist->title == NULL)
    title = filename;
  else
    title = thist->title;

  {
    const int not_active = 0;
    FILE *stream = util_fopen(filename,"w");
    int time_step;
    util_fwrite_string(title , stream);
    util_fwrite(&thist->data_size , sizeof thist->data_size , 1 , stream , __func__);
    for (time_step = 0; time_step < thist->data_size; time_step++) {
      if (thist->data[time_step] != NULL) 
	thist_node_matlab_dump(thist->data[time_step] , stream);
      else
	util_fwrite(&not_active , sizeof not_active , 1 , stream , __func__);
    }
    fclose(stream);
  }
  free(path);
}


void thist_clear(thist_type * thist) {
  int time_step;
  for (time_step = 0; time_step < thist->data_size; time_step++) 
    if (thist->data[time_step] != NULL) {
      thist_node_clear(thist->data[time_step]);
      thist->data[time_step] = NULL;
    }
  thist_set_title(thist , NULL);
}



