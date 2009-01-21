#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <util.h>
#include <ctype.h>
#include <menu.h>
#include <arg_pack.h>
#include <enkf_main.h>
#include <enkf_sched.h>
#include <enkf_ui_plot.h>
#include <enkf_ui_fs.h>
#include <enkf_obs.h>
#include <field_obs.h>
#include <field_config.h>
#include <obs_vector.h>
#include <plot.h>
#include <plot_dataset.h>
#include <enkf_ui_util.h>
#include <ensemble_config.h>
#include <msg.h>


static char * enkf_ui_plot_alloc_plot_file(const char * plot_path, const char * base_name) {
  const char * extension  =  "png";
  return util_alloc_filename( plot_path , base_name , extension);
}
					   
					   

static plot_type * __plot_alloc(const char * x_label , const char * y_label , const char * title , const char * file) {
  plot_type * plot  = plot_alloc();
  plot_set_window_size(plot , 640, 480);
  plot_initialize(plot , "png", file);
  plot_set_labels(plot, x_label , y_label , title);
  return plot;
}


static void __plot_add_data(plot_type * plot , int N , const double * x , const double *y) {
  plot_dataset_type *d = plot_alloc_new_dataset( plot , plot_xy , false);
  plot_dataset_set_line_color(d , BLUE);
  plot_dataset_append_vector_xy(d, N , x, y);
}


static void __plot_show(plot_type * plot , const char * viewer , const char * file) {
  plot_data(plot);
  plot_free(plot);
  util_vfork_exec(viewer , 1 , (const char *[1]) { file } , false , NULL , NULL , NULL , NULL , NULL);
}







static void enkf_ui_plot_ensemble__(enkf_fs_type * fs       , 
				    enkf_obs_type * enkf_obs, 
				    const enkf_config_node_type * config_node , 
				    const char * user_key  ,
				    const char * key_index ,
				    int step1 , int step2  , 
				    int iens1 , int iens2  , 
				    state_enum plot_state  ,
				    const char * plot_path , 
				    const char * viewer) {

  const bool add_observations = true;
  char * plot_file = enkf_ui_plot_alloc_plot_file( plot_path , user_key );
  plot_type * plot = __plot_alloc("x-akse","y-akse",user_key,plot_file);
  enkf_node_type * node;
  msg_type * msg;
  double *x , *y;
  int     size, iens , step;

  node = enkf_node_alloc( config_node );
  if (plot_state == both) 
    size = 2 * (step2 - step1 + 1);
  else
    size = (step2 - step1 + 1);

  x = util_malloc( size * sizeof * x, __func__);
  y = util_malloc( size * sizeof * y, __func__);
  {
    char * prompt = util_alloc_sprintf("Loading %s member/step: " , enkf_config_node_get_key(config_node));
    msg = msg_alloc(prompt);
    free(prompt);
  }
  msg_show(msg);
  
  for (iens = iens1; iens <= iens2; iens++) {
    char label[32];
	
    int this_size = 0;
    for (step = step1; step <= step2; step++) {
      sprintf(label , "%03d/%03d" , iens , step);
      msg_update( msg , label);
	  
      /* Forecast block */
      if (plot_state & forecast) {
	if (enkf_fs_has_node(fs , config_node , step , iens , forecast)) {
	  bool valid;
	  enkf_fs_fread_node(fs , node , step , iens , forecast);
	  y[this_size] = enkf_node_user_get( node , key_index , &valid);
	  if (valid) {
	    x[this_size] = step;
	    this_size++;
	  }
	} 
      }
	  
      /* Analyzed block */
      if (plot_state & analyzed) {
	if (enkf_fs_has_node(fs , config_node , step , iens , analyzed)) {
	  bool valid;
	  enkf_fs_fread_node(fs , node , step , iens , analyzed);
	  y[this_size] = enkf_node_user_get( node , key_index , &valid);
	  if (valid) {
	    x[this_size] = step;
	    this_size++;
	  }
	} 
      }
    }
    __plot_add_data(plot , this_size , x , y );
  }

  if (add_observations) {
    if (enkf_config_node_get_impl_type(config_node) == SUMMARY) {/* Adding observations only implemented for summary. */
      const stringlist_type * obs_keys = enkf_config_node_get_obs_keys(config_node);
      int i;
      for (i=0; i < stringlist_get_size( obs_keys ); i++) {
	const char * obs_key = stringlist_iget(obs_keys , i);
	plot_dataset_type * obs_data = plot_alloc_new_dataset( plot , plot_xy1y2 , false);
	const obs_vector_type * obs_vector = enkf_obs_get_vector( enkf_obs , obs_key);
	double  value , std;
	int report_step = -1;
	plot_dataset_set_line_color( obs_data , RED);
	plot_dataset_set_line_width( obs_data , 1.0);
	do {
	  report_step = obs_vector_get_next_active_step( obs_vector , report_step);
	  if (report_step != -1) {
	    if (report_step >= step1 && report_step <= step2) {
	      bool valid;
	      obs_vector_user_get( obs_vector , key_index , report_step , &value , &std , &valid);
	      if (valid)
		plot_dataset_append_point_xy1y2( obs_data , report_step , value - std , value + std);
	    }
	  }
	} while (report_step != -1);
      }
    }
  }
      
  plot_set_bottom_padding( plot , 0.05);
  plot_set_top_padding( plot    , 0.05);
  plot_set_left_padding( plot   , 0.05);
  plot_set_right_padding( plot  , 0.05);

  enkf_node_free(node);
  msg_free(msg , true);
  printf("Plot saved in: %s \n",plot_file);
  __plot_show(plot , viewer , plot_file); /* Frees the plot - logical ehhh. */
  free(plot_file);
}



void enkf_ui_plot_histogram(void * arg) {
  enkf_main_type             * enkf_main  = enkf_main_safe_cast( arg );
  const enkf_sched_type      * enkf_sched = enkf_main_get_enkf_sched(enkf_main);
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  enkf_fs_type               * fs              = enkf_main_get_fs(enkf_main);
  const char * viewer                          = enkf_main_get_image_viewer( enkf_main );
  const model_config_type    * model_config    = enkf_main_get_model_config( enkf_main );
  const char * plot_path                       = model_config_get_plot_path( model_config );
  {
    const int prompt_len = 40;
    const char * prompt  = "What do you want to plot (KEY:INDEX)";
    const enkf_config_node_type * config_node;
    char       user_key[64];
    
    
    util_printf_prompt(prompt , prompt_len , '=' , "=> ");
    scanf("%s" , user_key);
    {
      const int ens_size    = ensemble_config_get_size(ensemble_config);
      state_enum plot_state = analyzed; /* Compiler shut up */
      char * key_index;
      const int last_report = enkf_sched_get_last_report(enkf_sched);
      double * count        = util_malloc(ens_size * sizeof * count , __func__);
      int iens , report_step;
      char * plot_file = enkf_ui_plot_alloc_plot_file( plot_path , user_key );
      plot_type * plot = __plot_alloc("x-akse","y-akse",user_key,plot_file);

      config_node = ensemble_config_user_get_node( ensemble_config , user_key , &key_index);
      if (config_node == NULL) {
	fprintf(stderr,"** Sorry - could not find any nodes with the key:%s \n",user_key);
	util_safe_free(key_index);
	return;
      }
      report_step = util_scanf_int_with_limits("Report step: ", prompt_len , 0 , last_report);
      {
	enkf_var_type var_type = enkf_config_node_get_var_type(config_node);
	if ((var_type == dynamic_state) || (var_type == dynamic_result)) 
	  plot_state = enkf_ui_util_scanf_state("Plot Forecast/Analyzed: [F|A]" , prompt_len , false);
	else if (var_type == parameter)
	  plot_state = analyzed;
	else
	  util_abort("%s: can not plot this type \n",__func__);
      }
      {
	int active_size = 0;
	enkf_node_type * node = enkf_node_alloc( config_node );
	for (iens = 0; iens < ens_size; iens++) {
	  if (enkf_fs_has_node(fs , config_node , report_step , iens , plot_state)) {
	    bool valid;
	    enkf_fs_fread_node(fs , node , report_step , iens , forecast);
	    count[active_size] = enkf_node_user_get( node , key_index , &valid);
	    if (valid) 
	      active_size++;
	    
	  }
	}
	enkf_node_free( node );
	
	{
	  plot_dataset_type * d = plot_alloc_new_dataset( plot , plot_hist , true );
	  plot_dataset_set_shared_hist(d , active_size , count);
	}
	__plot_show(plot , viewer , plot_file);
      }
      free(count);
      util_safe_free(key_index);
    }
  }
}





void enkf_ui_plot_ensemble(void * arg) {
  enkf_main_type             * enkf_main  = enkf_main_safe_cast( arg );
  enkf_obs_type              * enkf_obs        = enkf_main_get_obs( enkf_main );
  const enkf_sched_type      * enkf_sched = enkf_main_get_enkf_sched(enkf_main);
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  enkf_fs_type               * fs              = enkf_main_get_fs(enkf_main);
  const char * viewer                          = enkf_main_get_image_viewer( enkf_main );
  const model_config_type    * model_config    = enkf_main_get_model_config( enkf_main );
  const char * plot_path                       = model_config_get_plot_path( model_config );
  {
    const int prompt_len = 40;
    const char * prompt  = "What do you want to plot (KEY:INDEX)";
    const enkf_config_node_type * config_node;
    char       user_key[64];
    
    
    util_printf_prompt(prompt , prompt_len , '=' , "=> ");
    scanf("%s" , user_key);
    {
      state_enum plot_state = analyzed; /* Compiler shut up */
      char * key_index;
      const int last_report = enkf_sched_get_last_report(enkf_sched);
      int iens1 , iens2 , step1 , step2;   
            
      config_node = ensemble_config_user_get_node( ensemble_config , user_key , &key_index);
      if (config_node == NULL) {
	fprintf(stderr,"** Sorry - could not find any nodes with the key:%s \n",user_key);
	util_safe_free(key_index);
	return;
      }

      enkf_ui_util_scanf_report_steps(last_report , prompt_len , &step1 , &step2);
      enkf_ui_util_scanf_iens_range(ensemble_config_get_size(ensemble_config) , prompt_len , &iens1 , &iens2);
      
      {
	enkf_var_type var_type = enkf_config_node_get_var_type(config_node);
	if ((var_type == dynamic_state) || (var_type == dynamic_result)) 
	  plot_state = enkf_ui_util_scanf_state("Plot Forecast/Analyzed/Both: [F|A|B]" , prompt_len , true);
	else if (var_type == parameter)
	  plot_state = analyzed;
	else
	  util_abort("%s: can not plot this type \n",__func__);
      }
      enkf_ui_plot_ensemble__(fs, 
			      enkf_obs,
			      config_node , 
			      user_key , 
			      key_index , 
			      step1 , 
			      step2 , 
			      iens1 , 
			      iens2 , 
			      plot_state , 
			      plot_path,
			      viewer);
      util_safe_free(key_index);
    }
  }
}
	
	   

void enkf_ui_plot_all_summary(void * arg) {
  enkf_main_type             * enkf_main       = enkf_main_safe_cast( arg );
  enkf_obs_type              * enkf_obs        = enkf_main_get_obs( enkf_main );
  const enkf_sched_type      * enkf_sched      = enkf_main_get_enkf_sched(enkf_main);
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  enkf_fs_type               * fs              = enkf_main_get_fs(enkf_main);
  const char * viewer                          = enkf_main_get_image_viewer( enkf_main );
  const model_config_type    * model_config    = enkf_main_get_model_config( enkf_main );
  const char * plot_path                       = model_config_get_plot_path( model_config );
  
  const int iens1        = 0;
  const int iens2        = ensemble_config_get_size(ensemble_config) - 1;
  const int last_report  = enkf_sched_get_last_report(enkf_sched);
  const int first_report = 0;
  
  {
    stringlist_type * summary_keys = ensemble_config_alloc_typed_keylist(ensemble_config , dynamic_result);
    int ikey;
    
    for (ikey = 0; ikey < stringlist_get_size( summary_keys ); ikey++) {
      const char * key = stringlist_iget( summary_keys , ikey);

      enkf_ui_plot_ensemble__(fs , 
			      enkf_obs , 
			      ensemble_config_get_node( ensemble_config , key ),
			      key , 
			      NULL , 
			      first_report , last_report , 
			      iens1 , iens2 , 
			      both  , 
			      plot_path , 
			      viewer);
      
    }
    stringlist_free( summary_keys );
  }
}


	  


void enkf_ui_plot_observation(void * arg) {
  enkf_main_type             * enkf_main       = enkf_main_safe_cast( arg );
  enkf_obs_type              * enkf_obs        = enkf_main_get_obs( enkf_main );
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  const char * viewer                          = enkf_main_get_image_viewer( enkf_main );
  const model_config_type    * model_config    = enkf_main_get_model_config( enkf_main );
  const char * plot_path                       = model_config_get_plot_path( model_config );
  {
    const int ens_size = ensemble_config_get_size(ensemble_config);
    const int prompt_len = 40;
    const char * prompt  = "What do you want to plot (KEY:INDEX)";
    enkf_fs_type   * fs   = enkf_main_get_fs(enkf_main);
    const obs_vector_type * obs_vector;
    char user_key[64];
    char * index_key;

    util_printf_prompt(prompt , prompt_len , '=' , "=> ");
    scanf("%s" , user_key);
    
    obs_vector = enkf_obs_user_get_vector(enkf_obs , user_key , &index_key);
    if (obs_vector != NULL) {
      char * plot_file                    = enkf_ui_plot_alloc_plot_file(plot_path , user_key);
      plot_type * plot                    = __plot_alloc("Member nr" , "Value" , user_key , plot_file);   
      const char * state_kw               = obs_vector_get_state_kw( obs_vector );
      enkf_config_node_type * config_node = ensemble_config_get_node( ensemble_config , state_kw );
      int   num_active                    = obs_vector_get_num_active( obs_vector );
      plot_dataset_type * obs_value       = plot_alloc_new_dataset(plot , plot_yline , false);
      plot_dataset_type * obs_quant       = plot_alloc_new_dataset(plot , plot_yline , false);
      plot_dataset_type * forecast_data   = plot_alloc_new_dataset(plot , plot_xy    , false);
      plot_dataset_type * analyzed_data   = plot_alloc_new_dataset(plot , plot_xy    , false);
      int   report_step;
      
      do {
	if (num_active == 1)
	  report_step = obs_vector_get_active_report_step( obs_vector );
	else
	  report_step = enkf_ui_util_scanf_report_step(enkf_main , "Report step" , prompt_len);
      } while (!obs_vector_iget_active(obs_vector , report_step));
      {
	enkf_node_type * enkf_node = enkf_node_alloc( config_node );
	msg_type * msg = msg_alloc("Loading realization: ");
	double y , value , std ;
	bool   valid;
	const int    iens1 = 0;
	const int    iens2 = ens_size - 1;
	int    iens;
	char  cens[5];

	obs_vector_user_get( obs_vector , index_key , report_step , &value , &std , &valid);
	plot_set_bottom_padding( plot , 0.10);
	plot_set_top_padding( plot , 0.10);
	plot_set_left_padding( plot , 0.05);
	plot_set_right_padding( plot , 0.05);
			    
	plot_dataset_append_point_yline(obs_value , value);
	plot_dataset_append_point_yline(obs_quant , value - std);
	plot_dataset_append_point_yline(obs_quant , value + std);
	
	plot_dataset_set_line_color(obs_value , BLACK);
	plot_dataset_set_line_color(obs_quant , BLACK);
	plot_dataset_set_line_width(obs_value , 2.0);
	plot_dataset_set_line_style(obs_quant , long_dash);

	plot_dataset_set_style( forecast_data , POINT);
	plot_dataset_set_style( analyzed_data , POINT);
	plot_dataset_set_point_color( forecast_data , BLUE );
	plot_dataset_set_point_color( analyzed_data , RED  );
	
	msg_show(msg);
	for (iens = iens1; iens <= iens2; iens++) {
	  sprintf(cens , "%03d" , iens);
	  msg_update(msg , cens);

	  if (enkf_fs_has_node(fs , config_node , report_step , iens , analyzed)) {
	    enkf_fs_fread_node(fs , enkf_node   , report_step , iens , analyzed);
	    y = enkf_node_user_get( enkf_node , index_key , &valid);
	    if (valid) 
	      plot_dataset_append_point_xy( analyzed_data , iens , y);
	  }

	  if (enkf_fs_has_node(fs , config_node , report_step , iens , forecast)) {
	    enkf_fs_fread_node(fs , enkf_node   , report_step , iens , forecast);
	    y = enkf_node_user_get( enkf_node , index_key , &valid);
	    if (valid) 
	      plot_dataset_append_point_xy( forecast_data , iens , y);
	  }
	  
	}
	msg_free(msg , true);
	printf("\n");
	enkf_node_free(enkf_node);
      }
      __plot_show(plot , viewer , plot_file);
      printf("Plot saved in: %s \n",plot_file);
      free(plot_file);
    } 
    
    util_safe_free( index_key );
  }
}



void enkf_ui_plot_RFT(void * arg) {
  enkf_main_type             * enkf_main       = enkf_main_safe_cast( arg );
  enkf_obs_type              * enkf_obs        = enkf_main_get_obs( enkf_main );
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  const char * viewer                          = enkf_main_get_image_viewer( enkf_main );
  const model_config_type    * model_config    = enkf_main_get_model_config( enkf_main );
  const char * plot_path                       = model_config_get_plot_path( model_config );
  {
    const int ens_size = ensemble_config_get_size(ensemble_config);
    const int prompt_len = 40;
    const char * prompt  = "Which RFT observation: ";
    enkf_fs_type   * fs   = enkf_main_get_fs(enkf_main);
    const obs_vector_type * obs_vector;
    char obs_key[64];
    char * plot_file;
    int   report_step;

    {
      bool OK = false;
      while (!OK) {
	util_printf_prompt(prompt , prompt_len , '=' , "=> ");
	scanf("%s" , obs_key);
	if (enkf_obs_has_key(enkf_obs , obs_key)) {
	  obs_vector = enkf_obs_get_vector( enkf_obs , obs_key );
	  if (obs_vector_get_impl_type( obs_vector ) == field_obs)
	    OK = true;
	  else
	    fprintf(stderr,"Observation key:%s does not correspond to a field observation.\n",obs_key);
	} else
	  fprintf(stderr,"Do not have observation key:%s \n",obs_key);
      }
    }
    do {
      if (obs_vector_get_num_active( obs_vector ) == 1)
	report_step = obs_vector_get_active_report_step( obs_vector );
      else
	report_step = enkf_ui_util_scanf_report_step(enkf_main , "Report step" , prompt_len);
    } while (!obs_vector_iget_active(obs_vector , report_step));
    
    /* OK - when we are here the user has entered a valid key which is a field observation. */
    {
      plot_type             * plot;
      const char            * state_kw    = obs_vector_get_state_kw(obs_vector);
      enkf_node_type        * node;
      enkf_config_node_type * config_node = ensemble_config_get_node( ensemble_config , state_kw );
      field_config_type * field_config    = enkf_config_node_get_ref( config_node );
      field_obs_type    * field_obs       = obs_vector_iget_node( obs_vector , report_step );
      //int   obs_size                      = field_obs_get_size( field_obs );
      
      
      
      
      plot_file = enkf_ui_plot_alloc_plot_file(plot_path , obs_key);
      plot = __plot_alloc(state_kw , "Depth" , obs_key , plot_file);
      {
	msg_type * msg       = msg_alloc("Loading realization: ");
	const int * i 	     = field_obs_get_i(field_obs);
	const int * j 	     = field_obs_get_j(field_obs);
	const int * k 	     = field_obs_get_k(field_obs);
	const int   obs_size = field_obs_get_size(field_obs);
	const ecl_grid_type * grid = field_config_get_grid( field_config );
	double * depth       = util_malloc( obs_size * sizeof * depth , __func__);
	double min_depth , max_depth;
	
	int l;
	int iens;
	int iens1 = 0;        /* Could be user input */
	int iens2 = ens_size;
	
	plot_dataset_type ** data = util_malloc( ens_size * sizeof * data , __func__);
	plot_dataset_type *  obs  = plot_alloc_new_dataset( plot , plot_x1x2y , false);
	node = enkf_node_alloc( config_node );
	
	for (l = 0; l < obs_size; l++) {
	  double xpos, ypos,zpos;
	  ecl_grid_get_pos(grid , i[l] , j[l] , k[l] , &xpos , &ypos , &zpos);
	  depth[l] = zpos;
	}
	
	max_depth = depth[0];
	min_depth = depth[0];
	for (l=1; l< obs_size; l++)
	  util_update_double_max_min( depth[l] , &max_depth , &min_depth);
	
	
	for (iens=iens1; iens <= iens2; iens++)
	  data[iens - iens1] = plot_alloc_new_dataset( plot , plot_xy , false);
	
	msg_show( msg );
	for (iens=iens1; iens <= iens2; iens++) {
	  char cens[5];
	  sprintf(cens , "%03d" , iens);
	  msg_update(msg , cens);

	  if (enkf_fs_has_node(fs , config_node , report_step , iens , analyzed)) {
	    enkf_fs_fread_node(fs , node , report_step , iens , analyzed);
	    const field_type * field = enkf_node_value_ptr( node );
	    for (l = 0; l < obs_size; l++)  /* l : kind of ran out of indices ... */
	      plot_dataset_append_point_xy(data[iens - iens1] , field_ijk_get_double( field , i[l] , j[l] , k[l]) , depth[l]);
	    
	  }
	}
	for (l = 0; l < obs_size; l++) {
	  double value , std;
	  
	  field_obs_iget(field_obs , l , &value , &std);
	  plot_dataset_append_point_x1x2y( obs , value - std , value + std , depth[l]);
	}

	plot_set_bottom_padding( plot , 0.05);
	plot_set_top_padding( plot , 0.05);
	plot_set_left_padding( plot , 0.05);
	plot_set_right_padding( plot , 0.05);
	plot_invert_y_axis( plot );
	
	plot_dataset_set_line_color( obs , RED );
	free(depth);
	msg_free(msg , true);
      }
      __plot_show( plot , viewer , plot_file);
      printf("Plot saved in: %s \n",plot_file);
      free(plot_file);
    }
  }
}






void enkf_ui_plot_menu(void * arg) {
  
  enkf_main_type  * enkf_main  = enkf_main_safe_cast( arg );  
  {
    const model_config_type * model_config = enkf_main_get_model_config( enkf_main );
    const char * plot_path  =  model_config_get_plot_path( model_config );
    util_make_path( plot_path );
  }

  {
    menu_type * menu = menu_alloc("EnKF plot menu" , "Back" , "bB");
    menu_add_item(menu , "Ensemble plot"    , "eE"                          , enkf_ui_plot_ensemble    , enkf_main , NULL);
    menu_add_item(menu , "Ensemble plot of ALL summary variables"    , "aA" , enkf_ui_plot_all_summary , enkf_main , NULL);
    menu_add_item(menu , "Observation plot" , "oO" 			    , enkf_ui_plot_observation , enkf_main , NULL);
    menu_add_item(menu , "RFT plot"         , "rR" 			    , enkf_ui_plot_RFT         , enkf_main , NULL);
    menu_add_item(menu , "Histogram"        , "hH"                          , enkf_ui_plot_histogram   , enkf_main , NULL);
    menu_add_separator(menu);
    menu_add_item(menu , "Change directories for reading and writing" , "cC" , enkf_ui_fs_menu , enkf_main , NULL);
    menu_run(menu);
    menu_free(menu);
  }
}
