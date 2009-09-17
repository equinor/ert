#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <util.h>
#include <ctype.h>
#include <menu.h>
#include <arg_pack.h>
#include <enkf_main.h>
#include <enkf_tui_plot.h>
#include <enkf_tui_fs.h>
#include <enkf_obs.h>
#include <field_obs.h>
#include <field_config.h>
#include <obs_vector.h>
#include <bool_vector.h>
#include <plot.h>
#include <plot_dataset.h>
#include <enkf_tui_util.h>
#include <ensemble_config.h>
#include <msg.h>
#include <vector.h>
#include <enkf_state.h>
#include <gen_kw_config.h>
#include <enkf_defaults.h>
#include <math.h>
#include <time.h>
#include <plot_config.h>


/**
   This vector of sched_file instances is used to translate from
   report_step to simulation days on the x-axis of the ensemble
   plots. This is a bit awkward for two reasons:

    * Most of the plot functions only relate to the ensemble_config
      and the enkf_fs filesystem; _not_ the enkf_state
      instances. These sched_file pointers break that premise. Should
      probably get the sim_time directly from the file-system?

    * The implementation supports the use of member-specific schedule
      files, and for this reason we must have member specific files
      her as well, this increases the complexity for something which
      is probably only used in 1/1000 cases.
      
*/

static vector_type * enkf_tui_alloc_sched_vector( const enkf_main_type * enkf_main ) {
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  const int ens_size                           = ensemble_config_get_size( ensemble_config );
  
  int iens;
  vector_type * vector = vector_alloc_new();
  for (iens = 0; iens < ens_size; iens++)
    vector_append_ref( vector , enkf_state_get_sched_file ( enkf_main_iget_state( enkf_main , iens )));
  
  return vector;
}



/**
   The final plot path consists of three parts: 

    plot_path: This is the PLOT_PATH option given in main configuration file.

    case_name: This the name of the currently active case.

    base_name: The filename of the current plot.
*/

static char * enkf_tui_plot_alloc_plot_file(const plot_config_type * plot_config , const char * case_name , const char * base_name) {
  char * path      = util_alloc_filename(plot_config_get_path( plot_config ) , case_name , NULL); /* It is really a path - but what the fuck. */ 
  char * plot_file = util_alloc_filename(path , base_name , plot_config_get_image_type( plot_config ));

  util_make_path( path );  /* Ensure that the path where the plots are stored exists. */
  free(path);
  return plot_file;
}
					   


static plot_type * __plot_alloc(const plot_config_type * plot_config , const char * x_label , const char * y_label , const char * title , const char * file) {
  
  arg_pack_type * arg_pack = arg_pack_alloc();
  plot_type * plot;
  
  if (util_string_equal( plot_config_get_driver( plot_config ) , "PLPLOT")) {
    arg_pack_append_ptr( arg_pack , file );
    arg_pack_append_ptr( arg_pack , plot_config_get_image_type( plot_config ));
  } else if (util_string_equal( plot_config_get_driver( plot_config ) , "TEXT")) {

    char * plot_path, *basename;
    char * path;
    printf("Splitting:%s \n",file);
    util_alloc_file_components( file , &plot_path , &basename , NULL);
    
    path = util_alloc_filename( plot_path , basename , NULL);
    arg_pack_append_owned_ptr( arg_pack , path , free);
    
    free( plot_path );
    free( basename );
  } else 
    util_abort("%s: unrecognized driver type: %s \n",__func__ , plot_config_get_driver( plot_config ));
  
  plot = plot_alloc(plot_config_get_driver( plot_config ) , arg_pack);
  
  plot_set_window_size(plot , plot_config_get_width( plot_config ) , plot_config_get_height( plot_config ));
  plot_set_labels(plot, x_label , y_label , title);
  arg_pack_free( arg_pack );
  
  return plot;
}


static void __plot_add_data(plot_type * plot , const char * label , int N , const double * x , const double *y) {
  plot_dataset_type *d = plot_alloc_new_dataset( plot , label , PLOT_XY );
  plot_dataset_set_line_color(d , BLUE);
  plot_dataset_append_vector_xy(d, N , x, y);
}


static void __plot_show(plot_type * plot , const plot_config_type * plot_config , const char * file) {
  plot_data(plot);
  plot_free(plot);
  if (util_file_exists( file )) {
    printf("Plot saved in: %s \n",file);
    util_vfork_exec(plot_config_get_viewer( plot_config ) , 1 , (const char *[1]) { file } , false , NULL , NULL , NULL , NULL , NULL);
  }
  /*
    else: the file does not exist - that might be OK?
  */
}







static void enkf_tui_plot_ensemble__(enkf_fs_type * fs       , 
                                     enkf_obs_type * enkf_obs, 
                                     const enkf_config_node_type * config_node , 
                                     const char * user_key  ,
                                     const char * key_index ,
                                     const vector_type * sched_vector , 
                                     int step1 , int step2  , 
                                     int iens1 , int iens2  , 
                                     state_enum plot_state  ,
                                     const plot_config_type * plot_config) {
  
  bool  plot_dates             = true;
  const int errorbar_max_obsnr = plot_config_get_errorbar_max( plot_config );
  const bool add_observations  = true;
  bool  show_plot              = false;
  char * plot_file = enkf_tui_plot_alloc_plot_file( plot_config , enkf_fs_get_read_dir(fs), user_key );
  plot_type * plot ;
  enkf_node_type * node;
  msg_type * msg;
  double *x , *y;
  bool_vector_type * has_data = bool_vector_alloc( step2 + 1 , false );
  int     size, iens , step;

  if (plot_dates)
    plot =  __plot_alloc(plot_config , "" , /* y akse */ "" ,user_key,plot_file);
  else
    plot =  __plot_alloc(plot_config , "Simulation time (days) ", /* y akse */ "" ,user_key , plot_file);
  
  node = enkf_node_alloc( config_node );
  if (plot_state == BOTH) 
    size = 2 * (step2 - step1 + 1);
  else
    size = (step2 - step1 + 1);

  x = util_malloc( size * sizeof * x, __func__);
  y = util_malloc( size * sizeof * y, __func__);
  {
    char * prompt = util_alloc_sprintf("Loading %s member: " , enkf_config_node_get_key(config_node));
    msg = msg_alloc(prompt);
    free(prompt);
  }
  msg_show(msg);
  

  for (iens = iens1; iens <= iens2; iens++) {
    char msg_label[32];
    char plot_label[32];
    int this_size = 0;
    sprintf(msg_label , "%03d" , iens );
    msg_update( msg , msg_label);
    for (step = step1; step <= step2; step++) {
      double sim_days = sched_file_get_sim_days( vector_iget( sched_vector , iens) , step );
      time_t sim_time = sched_file_get_sim_time( vector_iget( sched_vector , iens) , step );
	  
      /* Forecast block */
      if (plot_state & FORECAST) {
	if (enkf_fs_has_node(fs , config_node , step , iens , FORECAST)) {
	  bool valid;
	  enkf_fs_fread_node(fs , node , step , iens , FORECAST);
	  y[this_size] = enkf_node_user_get( node , key_index , &valid);

	  if ((iens == 2) && (step == 10))
	    y[this_size] = NAN;
	  
	  if ((iens == 2) && (step == 20))
	    y[this_size] = INFINITY;

	  bool_vector_iset(has_data , step , true);
	  if (valid) {
	    if (plot_dates)
	      x[this_size] = sim_time;
	    else
	      x[this_size] = sim_days;
	    this_size++;
	  }
	} 
      }
	  
      /* Analyzed block */
      if (plot_state & ANALYZED) {
	if (enkf_fs_has_node(fs , config_node , step , iens , ANALYZED)) {
	  bool valid;
	  enkf_fs_fread_node(fs , node , step , iens , ANALYZED);
	  y[this_size] = enkf_node_user_get( node , key_index , &valid);
	  bool_vector_iset(has_data , step , true);
	  if (valid) {
	    if (plot_dates)
	      x[this_size] = sim_time;
	    else
	      x[this_size] = sim_days;
	    this_size++;
	  }
	} 
      }
    }
    if (this_size > 0)
      show_plot = true;
    
    if (plot_dates) 
      plot_set_default_timefmt( plot , (time_t) x[0] , (time_t) x[this_size - 1]);
    
    sprintf(plot_label , "mem_%03d" , iens);
    __plot_add_data(plot , plot_label ,this_size , x , y );
  }


  /*
    Observe that all the observations are 'flattened'.
  */
  if (add_observations) {
    enkf_impl_type impl_type = enkf_config_node_get_impl_type(config_node);
    if ((impl_type == SUMMARY) || (impl_type == FIELD)) {
      /*
	These three double vectors are used to assemble
	all observations.
      */
      double_vector_type * sim_time     = double_vector_alloc( 0 , 0 );
      double_vector_type * obs_value    = double_vector_alloc( 0 , 0 );
      double_vector_type * obs_std      = double_vector_alloc( 0 , 0 );

      const stringlist_type * obs_keys  = enkf_config_node_get_obs_keys(config_node);
      int obs_size = 0;
      int i;
      for (i=0; i < stringlist_get_size( obs_keys ); i++) {
	const char * obs_key = stringlist_iget(obs_keys , i);
	const obs_vector_type * obs_vector = enkf_obs_get_vector( enkf_obs , obs_key);
	obs_size += obs_vector_get_num_active( obs_vector );
      }
      
      for (i=0; i < stringlist_get_size( obs_keys ); i++) {
	const char * obs_key = stringlist_iget(obs_keys , i);
	
	const obs_vector_type * obs_vector = enkf_obs_get_vector( enkf_obs , obs_key);
	double  value , std;
	int report_step = -1;
	do {
	  report_step = obs_vector_get_next_active_step( obs_vector , report_step);
	  if (report_step != -1) {
	    if (bool_vector_safe_iget( has_data , report_step)) {   /* Not plotting an observation if we do not have any simulations at the same time. */
	      bool valid;
	      obs_vector_user_get( obs_vector , key_index , report_step , &value , &std , &valid);
	      if (valid) {

		if (plot_dates)
		  double_vector_append( sim_time  , sched_file_get_sim_time( vector_iget( sched_vector , iens1) , report_step ));  
		else
		  double_vector_append( sim_time  , sched_file_get_sim_days( vector_iget( sched_vector , iens1) , report_step ));
		
		double_vector_append( obs_value , value );
		double_vector_append( obs_std , std );
	      }
	    }
	  }
	} while (report_step != -1);
      }

      if (double_vector_size( sim_time ) > 0) {
	if (obs_size > errorbar_max_obsnr) {
	  /* 
	     There are very many observations - to increase
	     readability of the plots we plot an upper and lower limit
	     as dashed lines, instead of plotting each individual
	     error bar.
	  */
	     
	  plot_dataset_type * data_value = plot_alloc_new_dataset( plot , "observation"       , PLOT_XY );
	  plot_dataset_type * data_lower = plot_alloc_new_dataset( plot , "observation_lower" , PLOT_XY );
	  plot_dataset_type * data_upper = plot_alloc_new_dataset( plot , "observation_upper" , PLOT_XY );
	  
	  plot_dataset_set_style( data_value , POINTS );
	  plot_dataset_set_style( data_upper , LINE );
	  plot_dataset_set_style( data_lower , LINE );
	  
	  plot_dataset_set_line_style( data_upper , PLOT_LINESTYLE_LONG_DASH );
	  plot_dataset_set_line_style( data_lower , PLOT_LINESTYLE_LONG_DASH );
	  plot_dataset_set_line_color( data_upper , RED);
	  plot_dataset_set_line_color( data_lower , RED);
	  plot_dataset_set_line_width( data_upper , 1.50 );
	  plot_dataset_set_line_width( data_lower , 1.50 );

	  plot_dataset_set_point_color( data_value , BLACK);
	  plot_dataset_set_symbol_type( data_value , PLOT_SYMBOL_FILLED_CIRCLE);

	  {
	    int * perm = double_vector_alloc_sort_perm( sim_time );
	    double_vector_permute( sim_time  , perm );
	    double_vector_permute( obs_value , perm );
	    double_vector_permute( obs_std   , perm );
	    free( perm );
	  }
	  
	  for (i = 0; i < double_vector_size( sim_time ); i++) {
	    double days  = double_vector_iget( sim_time  , i);
	    double value = double_vector_iget( obs_value , i);
	    double std   = double_vector_iget( obs_std   , i);
	    
	    plot_dataset_append_point_xy( data_value , days , value);
	    plot_dataset_append_point_xy( data_lower , days , value - std);
	    plot_dataset_append_point_xy( data_upper , days , value + std);
	  }
	} else {
	  plot_dataset_type * obs_errorbar  = plot_alloc_new_dataset( plot , "observations" , PLOT_XY1Y2 );
	  plot_dataset_set_line_color( obs_errorbar , RED);
	  plot_dataset_set_line_width( obs_errorbar , 1.5);
	  for (i = 0; i < double_vector_size( sim_time ); i++) {
	    double days  = double_vector_iget( sim_time  , i);
	    double value = double_vector_iget( obs_value , i);
	    double std   = double_vector_iget( obs_std   , i);
	    plot_dataset_append_point_xy1y2( obs_errorbar , days , value - std , value + std);
	  }
	}
      }
      double_vector_free( sim_time );
      double_vector_free( obs_std );
      double_vector_free( obs_value );
    }
  }

  plot_set_bottom_padding( plot , 0.05);
  plot_set_top_padding( plot    , 0.05);
  plot_set_left_padding( plot   , 0.05);
  plot_set_right_padding( plot  , 0.05);

  enkf_node_free(node);
  msg_free(msg , true);
  if (show_plot) 
    __plot_show(plot , plot_config , plot_file); /* Frees the plot - logical ehhh. */
  else {
    printf("No data to plot \n");
    plot_free(plot);
  }
	   
  free(plot_file);
  bool_vector_free( has_data );
}



void enkf_tui_plot_GEN_KW__(enkf_main_type * enkf_main , const enkf_config_node_type * config_node , int step1 , int step2 , int iens1 , int iens2 , vector_type * sched_vector) {
  enkf_fs_type               * fs           = enkf_main_get_fs(enkf_main);
  enkf_obs_type              * enkf_obs     = enkf_main_get_obs( enkf_main );
  const plot_config_type     * plot_config  = enkf_main_get_plot_config( enkf_main );
  gen_kw_config_type * gen_kw_config 	    = enkf_config_node_get_ref( config_node );
  int num_kw                         	    = gen_kw_config_get_data_size( gen_kw_config );
  const char ** key_list             	    = gen_kw_config_get_name_list( gen_kw_config );
  
  int ikw;
  
  for (ikw = 0; ikw < num_kw; ikw++) {
    char * user_key = gen_kw_config_alloc_user_key( gen_kw_config , ikw );
    enkf_tui_plot_ensemble__( fs , enkf_obs , config_node , user_key , key_list[ikw] , sched_vector,
                              step1 , step2 , iens1 , iens2 , ANALYZED , plot_config );
    free( user_key );
  }
}


void enkf_tui_plot_GEN_KW(void * arg) {
  enkf_main_type             * enkf_main       = enkf_main_safe_cast( arg );
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  {
    const int prompt_len = 50;
    const char * prompt  = "Which GEN_KW parameter do you want to plot";
    const enkf_config_node_type * config_node = NULL;
    bool  exit_loop = false;

    do {
      char *node_key;
      util_printf_prompt(prompt , prompt_len , '=' , "=> ");
      node_key = util_alloc_stdin_line();

      if (node_key != NULL) {
	if (!ensemble_config_has_key( ensemble_config , node_key ))
	  printf("Could not find node:%s \n",node_key);
	else {
	  config_node = ensemble_config_get_node( ensemble_config , node_key );
	  if (enkf_config_node_get_impl_type( config_node ) == GEN_KW) 
	    exit_loop = true;
	  else {
	    printf("%s is not a GEN_KW parameter \n",node_key);
	    config_node = NULL;
	  }
	}
      } else
	exit_loop = true;
      util_safe_free( node_key );
    } while (!exit_loop);

    if (config_node != NULL) {
      int iens1 , iens2 , step1 , step2;   
      vector_type * sched_vector = enkf_tui_alloc_sched_vector( enkf_main );
      const int last_report      = enkf_main_get_total_length( enkf_main );

      enkf_tui_util_scanf_report_steps(last_report , prompt_len , &step1 , &step2);
      enkf_tui_util_scanf_iens_range("Realizations members to plot(0 - %d)" , ensemble_config_get_size(ensemble_config) , prompt_len , &iens1 , &iens2);
      
      enkf_tui_plot_GEN_KW__(enkf_main , config_node , step1 , step2 , iens1 , iens2 , sched_vector);
      vector_free( sched_vector );
    }
  }
}

			 

void enkf_tui_plot_all_GEN_KW(void * arg) {
  enkf_main_type             * enkf_main       = enkf_main_safe_cast( arg );
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  {
    const int prompt_len = 40;
    int iens1 , iens2 , step1 , step2 , ikey;   
    vector_type * sched_vector    = enkf_tui_alloc_sched_vector( enkf_main );
    stringlist_type * gen_kw_keys = ensemble_config_alloc_keylist_from_impl_type(ensemble_config , GEN_KW);
    const int last_report         = enkf_main_get_total_length( enkf_main );

    enkf_tui_util_scanf_report_steps(last_report , prompt_len , &step1 , &step2);
    enkf_tui_util_scanf_iens_range("Realizations members to plot(0 - %d)" , ensemble_config_get_size(ensemble_config) , prompt_len , &iens1 , &iens2);
    
    for (ikey = 0; ikey < stringlist_get_size( gen_kw_keys ); ikey++) {
      const char * key = stringlist_iget( gen_kw_keys , ikey);
      enkf_config_node_type * config_node = ensemble_config_get_node( ensemble_config , key ); 
      enkf_tui_plot_GEN_KW__(enkf_main , config_node , step1 , step2 , iens1 , iens2 , sched_vector);
    }
    vector_free( sched_vector );
    stringlist_free( gen_kw_keys );
  }
}





void enkf_tui_plot_histogram(void * arg) {
  enkf_main_type             * enkf_main  = enkf_main_safe_cast( arg );
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  enkf_fs_type               * fs              = enkf_main_get_fs(enkf_main);
  const plot_config_type     * plot_config     = enkf_main_get_plot_config( enkf_main );
  const char                 * case_name       = enkf_fs_get_read_dir( fs );     
  {
    const int prompt_len = 40;
    const char * prompt  = "What do you want to plot (KEY:INDEX)";
    const enkf_config_node_type * config_node;
    char       * user_key;
    
    
    util_printf_prompt(prompt , prompt_len , '=' , "=> ");
    user_key = util_alloc_stdin_line();
    if (user_key != NULL) {
      const int ens_size    = ensemble_config_get_size(ensemble_config);
      state_enum plot_state = ANALYZED; /* Compiler shut up */
      char * key_index;
      const int last_report = enkf_main_get_total_length( enkf_main );
      double * count        = util_malloc(ens_size * sizeof * count , __func__);
      int iens , report_step;
      char * plot_file = enkf_tui_plot_alloc_plot_file( plot_config , case_name , user_key );
      plot_type * plot = __plot_alloc(plot_config , user_key , "#" ,user_key , plot_file);

      config_node = ensemble_config_user_get_node( ensemble_config , user_key , &key_index);
      if (config_node == NULL) {
	fprintf(stderr,"** Sorry - could not find any nodes with the key:%s \n",user_key);
	util_safe_free(key_index);
	return;
      }
      report_step = util_scanf_int_with_limits("Report step: ", prompt_len , 0 , last_report);
      {
	enkf_var_type var_type = enkf_config_node_get_var_type(config_node);
	if ((var_type == DYNAMIC_STATE) || (var_type == DYNAMIC_RESULT)) 
	  plot_state = enkf_tui_util_scanf_state("Plot Forecast/Analyzed: [F|A]" , prompt_len , false);
	else if (var_type == PARAMETER)
	  plot_state = ANALYZED;
	else
	  util_abort("%s: can not plot this type \n",__func__);
      }
      {
	int active_size = 0;
	enkf_node_type * node = enkf_node_alloc( config_node );
	for (iens = 0; iens < ens_size; iens++) {
	  if (enkf_fs_has_node(fs , config_node , report_step , iens , plot_state)) {
	    bool valid;
	    enkf_fs_fread_node(fs , node , report_step , iens , FORECAST);
	    count[active_size] = enkf_node_user_get( node , key_index , &valid);
	    if (valid) 
	      active_size++;
	    
	  }
	}
	enkf_node_free( node );
	
	{
	  plot_dataset_type * d = plot_alloc_new_dataset( plot , NULL , PLOT_HIST);
	  plot_dataset_append_vector_hist(d , active_size , count);
	}
        
	__plot_show(plot , plot_config , plot_file);
      }
      free(count);
      util_safe_free(key_index);
    }
    util_safe_free( user_key );
  }
}





void enkf_tui_plot_ensemble(void * arg) {
  enkf_main_type             * enkf_main       = enkf_main_safe_cast( arg );
  enkf_obs_type              * enkf_obs        = enkf_main_get_obs( enkf_main );
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  enkf_fs_type               * fs              = enkf_main_get_fs(enkf_main);
  const plot_config_type     * plot_config     = enkf_main_get_plot_config( enkf_main );
  vector_type * sched_vector                   = enkf_tui_alloc_sched_vector( enkf_main );
  {
    const int prompt_len = 40;
    const char * prompt  = "What do you want to plot (KEY:INDEX)";
    const enkf_config_node_type * config_node;
    char * user_key;
    
    util_printf_prompt(prompt , prompt_len , '=' , "=> ");
    user_key = util_alloc_stdin_line();
    if (user_key != NULL) {
      state_enum plot_state = ANALYZED; /* Compiler shut up */
      char * key_index;
      const int last_report = enkf_main_get_total_length( enkf_main );
      int iens1 , iens2 , step1 , step2;   
            
      config_node = ensemble_config_user_get_node( ensemble_config , user_key , &key_index);
      if (config_node == NULL) {
	fprintf(stderr,"** Sorry - could not find any nodes with the key:%s \n",user_key);
	util_safe_free(key_index);
	return;
      }

      enkf_tui_util_scanf_report_steps(last_report , prompt_len , &step1 , &step2);
      enkf_tui_util_scanf_iens_range("Realizations members to plot(0 - %d)" , ensemble_config_get_size(ensemble_config) , prompt_len , &iens1 , &iens2);
      
      {
	enkf_var_type var_type = enkf_config_node_get_var_type(config_node);
	if ((var_type == DYNAMIC_STATE) || (var_type == DYNAMIC_RESULT)) 
	  plot_state = enkf_tui_util_scanf_state("Plot Forecast/Analyzed/Both: [F|A|B]" , prompt_len , true);
	else if (var_type == PARAMETER)
	  plot_state = ANALYZED;
	else
	  util_abort("%s: can not plot this type \n",__func__);
      }
      enkf_tui_plot_ensemble__(fs, 
                               enkf_obs,
                               config_node , 
                               user_key , 
                               key_index , 
                               sched_vector , 
                               step1 , 
                               step2 , 
                               iens1 , 
                               iens2 , 
                               plot_state , 
                               plot_config);
      util_safe_free(key_index);
    }
    util_safe_free( user_key );
  }
  vector_free( sched_vector );
}
	
	   

void enkf_tui_plot_all_summary(void * arg) {
  enkf_main_type             * enkf_main       = enkf_main_safe_cast( arg );
  enkf_obs_type              * enkf_obs        = enkf_main_get_obs( enkf_main );
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  enkf_fs_type               * fs              = enkf_main_get_fs(enkf_main);
  const plot_config_type     * plot_config     = enkf_main_get_plot_config( enkf_main );
  vector_type * sched_vector                   = enkf_tui_alloc_sched_vector( enkf_main );  
  int last_report                              = enkf_main_get_total_length( enkf_main );
  const int prompt_len = 40;
  int iens1 , iens2 , step1 , step2;   

        
  enkf_tui_util_scanf_report_steps(last_report , prompt_len , &step1 , &step2);
  enkf_tui_util_scanf_iens_range("Realizations members to plot(0 - %d)" , ensemble_config_get_size(ensemble_config) , prompt_len , &iens1 , &iens2);
  
  {
    stringlist_type * summary_keys = ensemble_config_alloc_keylist_from_impl_type(ensemble_config , SUMMARY);
    int ikey;
    for (ikey = 0; ikey < stringlist_get_size( summary_keys ); ikey++) {
      const char * key = stringlist_iget( summary_keys , ikey);
      
      enkf_tui_plot_ensemble__(fs , 
                               enkf_obs , 
                               ensemble_config_get_node( ensemble_config , key ),
                               key , 
                               NULL , 
                               sched_vector , 
                               step1 , step2 , 
                               iens1 , iens2 , 
                               BOTH  , 
                               plot_config);
      
    }
    stringlist_free( summary_keys );
  }
  vector_free( sched_vector );
}


	  


void enkf_tui_plot_observation(void * arg) {
  enkf_main_type             * enkf_main       = enkf_main_safe_cast( arg );
  enkf_obs_type              * enkf_obs        = enkf_main_get_obs( enkf_main );
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  const plot_config_type     * plot_config     = enkf_main_get_plot_config( enkf_main );
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
      char * plot_file                    = enkf_tui_plot_alloc_plot_file(plot_config , enkf_fs_get_read_dir(fs), user_key);
      plot_type * plot                    = __plot_alloc(plot_config , "Member nr" , "Value" , user_key , plot_file);   
      const char * state_kw               = obs_vector_get_state_kw( obs_vector );
      enkf_config_node_type * config_node = ensemble_config_get_node( ensemble_config , state_kw );
      int   num_active                    = obs_vector_get_num_active( obs_vector );
      plot_dataset_type * obs_value       = plot_alloc_new_dataset(plot , "observation"   , PLOT_YLINE );
      plot_dataset_type * obs_quant_lower = plot_alloc_new_dataset(plot , "obs_minus_std" , PLOT_YLINE );
      plot_dataset_type * obs_quant_upper = plot_alloc_new_dataset(plot , "obs_plus_std"  , PLOT_YLINE );
      plot_dataset_type * forecast_data   = plot_alloc_new_dataset(plot , "forecast"      , PLOT_XY    );
      plot_dataset_type * analyzed_data   = plot_alloc_new_dataset(plot , "analyzed"      , PLOT_XY    );
      int   report_step;
      
      do {
	if (num_active == 1)
	  report_step = obs_vector_get_active_report_step( obs_vector );
	else
	  report_step = enkf_tui_util_scanf_report_step(enkf_main_get_total_length( enkf_main ) , "Report step" , prompt_len);
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
			    
	plot_dataset_set_yline(obs_value       , value);
	plot_dataset_set_yline(obs_quant_lower , value - std);
	plot_dataset_set_yline(obs_quant_upper , value + std);
	
	plot_dataset_set_line_color(obs_value       , BLACK);
	plot_dataset_set_line_color(obs_quant_lower , BLACK);
	plot_dataset_set_line_color(obs_quant_upper , BLACK);
	plot_dataset_set_line_width(obs_value , 2.0);
	plot_dataset_set_line_style(obs_quant_lower , PLOT_LINESTYLE_LONG_DASH);
        plot_dataset_set_line_style(obs_quant_upper , PLOT_LINESTYLE_LONG_DASH);



	plot_dataset_set_style( forecast_data , POINTS);
	plot_dataset_set_style( analyzed_data , POINTS);
	plot_dataset_set_point_color( forecast_data , BLUE );
	plot_dataset_set_point_color( analyzed_data , RED  );
	
	msg_show(msg);
	for (iens = iens1; iens <= iens2; iens++) {
	  sprintf(cens , "%03d" , iens);
	  msg_update(msg , cens);

	  if (enkf_fs_has_node(fs , config_node , report_step , iens , ANALYZED)) {
	    enkf_fs_fread_node(fs , enkf_node   , report_step , iens , ANALYZED);
	    y = enkf_node_user_get( enkf_node , index_key , &valid);
	    if (valid) 
	      plot_dataset_append_point_xy( analyzed_data , iens , y);
	  }

	  if (enkf_fs_has_node(fs , config_node , report_step , iens , FORECAST)) {
	    enkf_fs_fread_node(fs , enkf_node   , report_step , iens , FORECAST);
	    y = enkf_node_user_get( enkf_node , index_key , &valid);
	    if (valid) 
	      plot_dataset_append_point_xy( forecast_data , iens , y);
	  }
	  
	}
	msg_free(msg , true);
	printf("\n");
	enkf_node_free(enkf_node);
      }
      __plot_show(plot , plot_config , plot_file);
      free(plot_file);
    } 
    
    util_safe_free( index_key );
  }
}


void enkf_tui_plot_RFT__(enkf_fs_type * fs, const plot_config_type * plot_config , const model_config_type * model_config , const ensemble_config_type * ensemble_config , const obs_vector_type * obs_vector , const char * obs_key , int report_step) {
  plot_type             * plot;
  const char            * state_kw    = obs_vector_get_state_kw(obs_vector);
  enkf_node_type        * node;
  const int ens_size                  = ensemble_config_get_size(ensemble_config);
  enkf_config_node_type * config_node = ensemble_config_get_node( ensemble_config , state_kw );
  field_config_type * field_config    = enkf_config_node_get_ref( config_node );
  field_obs_type    * field_obs       = obs_vector_iget_node( obs_vector , report_step );
  char * plot_file;
  
  plot_file = enkf_tui_plot_alloc_plot_file(plot_config , enkf_fs_get_read_dir(fs), obs_key );
  plot = __plot_alloc(plot_config , state_kw , "Depth" , obs_key , plot_file);
  {
    msg_type * msg             = msg_alloc("Loading realization: ");
    const int * i 	       = field_obs_get_i(field_obs);
    const int * j 	       = field_obs_get_j(field_obs);
    const int * k 	       = field_obs_get_k(field_obs);
    const int   obs_size       = field_obs_get_size(field_obs);
    const ecl_grid_type * grid = field_config_get_grid( field_config );
    double * depth             = util_malloc( obs_size * sizeof * depth , __func__);
    double min_depth , max_depth;
    
    int l;
    int iens;
    int iens1 = 0;        /* Could be user input */
    int iens2 = ens_size - 1;
    
    plot_dataset_type *  obs;
    node = enkf_node_alloc( config_node );
    
    for (l = 0; l < obs_size; l++) {
      double xpos, ypos,zpos;
      ecl_grid_get_pos3(grid , i[l] , j[l] , k[l] , &xpos , &ypos , &zpos);
      depth[l] = zpos;
    }
    
    max_depth = depth[0];
    min_depth = depth[0];
    for (l=1; l< obs_size; l++)
      util_update_double_max_min( depth[l] , &max_depth , &min_depth);
    
    
    msg_show( msg );
    for (iens=iens1; iens <= iens2; iens++) {
      char cens[5];
      sprintf(cens , "%03d" , iens);
      msg_update(msg , cens);
      bool has_node = true;

      if (enkf_fs_has_node(fs , config_node , report_step , iens , ANALYZED)) /* Trying analyzed first. */
	enkf_fs_fread_node(fs , node , report_step , iens , ANALYZED);
      else if (enkf_fs_has_node(fs , config_node , report_step , iens , FORECAST))
	enkf_fs_fread_node(fs , node , report_step , iens , FORECAST);
      else 
	has_node = false;
      
      if (has_node) {
	const field_type * field = enkf_node_value_ptr( node );
	plot_dataset_type * data = plot_alloc_new_dataset( plot , NULL , PLOT_XY);
	plot_dataset_set_style( data , POINTS );
	plot_dataset_set_symbol_size( data , 1.00 );
	for (l = 0; l < obs_size; l++)  /* l : kind of ran out of indices ... */
	  plot_dataset_append_point_xy(data , field_ijk_get_double( field , i[l] , j[l] , k[l]) , depth[l]);
      } else printf("No data found for :%d/%d \n",iens, report_step);
    }
    
    obs = plot_alloc_new_dataset( plot , NULL , PLOT_X1X2Y );
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
  __plot_show( plot , plot_config , plot_file);
  printf("Plot saved in: %s \n",plot_file);
  free(plot_file);
}


static void enkf_tui_plot_select_RFT(const enkf_main_type * enkf_main , char ** _obs_key , int * _report_step) {
  enkf_obs_type              * enkf_obs        = enkf_main_get_obs( enkf_main );
  {
    const int prompt_len = 40;
    const char * prompt  = "Which RFT observation: ";

    const obs_vector_type * obs_vector;
    char  *obs_key;
    int    report_step;
    {
      bool OK = false;
      while (!OK) {
	util_printf_prompt(prompt , prompt_len , '=' , "=> ");
	obs_key = util_alloc_stdin_line( );
	if (enkf_obs_has_key(enkf_obs , obs_key)) {
	  obs_vector = enkf_obs_get_vector( enkf_obs , obs_key );
	  if (obs_vector_get_impl_type( obs_vector ) == FIELD_OBS)
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
	report_step = enkf_tui_util_scanf_report_step(enkf_main_get_total_length( enkf_main ) , "Report step" , prompt_len);
    } while (!obs_vector_iget_active(obs_vector , report_step));
    *_obs_key = obs_key;
    *_report_step = report_step;
  }
}



void enkf_tui_plot_RFT_depth(void * arg) {
  enkf_main_type             * enkf_main       = enkf_main_safe_cast( arg );
  enkf_obs_type              * enkf_obs        = enkf_main_get_obs( enkf_main );
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  const model_config_type    * model_config    = enkf_main_get_model_config( enkf_main );
  const plot_config_type    * plot_config      = enkf_main_get_plot_config( enkf_main );
  enkf_fs_type   * fs                          = enkf_main_get_fs(enkf_main);    
  {
    char * obs_key;
    int report_step;
    obs_vector_type * obs_vector;
    
    enkf_tui_plot_select_RFT(enkf_main , &obs_key , &report_step);
    obs_vector = enkf_obs_get_vector( enkf_obs , obs_key );
    enkf_tui_plot_RFT__(fs , plot_config , model_config , ensemble_config , obs_vector , obs_key , report_step);
    free( obs_key );
    
  }
}



void enkf_tui_plot_RFT_time(void * arg) {
  enkf_main_type             * enkf_main       = enkf_main_safe_cast( arg );
  enkf_obs_type              * enkf_obs        = enkf_main_get_obs( enkf_main );
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  const plot_config_type * plot_config         = enkf_main_get_plot_config(enkf_main);
  
  enkf_fs_type   * fs                          = enkf_main_get_fs(enkf_main);    
  vector_type * sched_vector                   = enkf_tui_alloc_sched_vector( enkf_main );
  {
    const char * state_kw;
    char * index_key = NULL;
    char * user_key  = NULL;
    char * obs_key;
    int report_step;
    obs_vector_type       * obs_vector;
    enkf_config_node_type * config_node;
    int step1 , step2;
    int iens1 , iens2;
    state_enum plot_state;

    enkf_tui_plot_select_RFT(enkf_main , &obs_key , &report_step);
    obs_vector  = enkf_obs_get_vector( enkf_obs , obs_key );
    config_node = obs_vector_get_config_node( obs_vector );

    /* Could be user input ... */
    step1      = 0;
    step2      = enkf_main_get_total_length( enkf_main );
    iens1      = 0;
    iens2      = ensemble_config_get_size(ensemble_config) - 1;
    plot_state = BOTH;
    state_kw   = enkf_config_node_get_key( config_node );
    {
      int block_nr,i,j,k;
      const field_obs_type * field_obs = obs_vector_iget_node( obs_vector , report_step );
      for (block_nr = 0; block_nr < field_obs_get_size( field_obs ); block_nr++) {
	field_obs_iget_ijk( field_obs , block_nr , &i , &j , &k);
	index_key = util_realloc_sprintf( index_key , "%d,%d,%d"    , i+1,j+1,k+1);
	user_key  = util_realloc_sprintf( user_key  , "%s:%d,%d,%d" , state_kw , i+1,j+1,k+1);
	enkf_tui_plot_ensemble__(fs , enkf_obs , config_node , user_key , index_key , sched_vector , step1 , step2 , iens1 , iens2 , plot_state , plot_config );
      }
    }
    free( obs_key );
    free( index_key );
    free( user_key );
  }
}




/**
   This function plots all the RFT's - observe that 'RFT' is no
   fundamental type in the enkf_obs type system. It will plot all
   BLOCK_OBS observations, they will typically (99% ??) be Pressure
   observations, but could in principle also be saturation observatioons.
*/



void enkf_tui_plot_all_RFT( void * arg) {
  enkf_main_type             * enkf_main       = enkf_main_safe_cast( arg );
  enkf_obs_type              * enkf_obs        = enkf_main_get_obs( enkf_main );
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  const model_config_type    * model_config    = enkf_main_get_model_config( enkf_main );
  const plot_config_type    * plot_config      = enkf_main_get_plot_config( enkf_main );
  {
    const int prompt_len  = 30;
    enkf_fs_type   * fs   = enkf_main_get_fs(enkf_main);
    int iobs , report_step;
    stringlist_type * RFT_keys = enkf_obs_alloc_typed_keylist(enkf_obs , FIELD_OBS);
    
    for (iobs = 0; iobs < stringlist_get_size( RFT_keys ); iobs++) {
      const char * obs_key = stringlist_iget( RFT_keys , iobs);
      const obs_vector_type * obs_vector = enkf_obs_get_vector( enkf_obs , obs_key );
      
      do {
	if (obs_vector_get_num_active( obs_vector ) == 1)
	  report_step = obs_vector_get_active_report_step( obs_vector );
	else 
	  /* An RFT should really be active at only one report step - but ... */
	  report_step = enkf_tui_util_scanf_report_step(enkf_main_get_total_length( enkf_main ) , "Report step" , prompt_len);
      } while (!obs_vector_iget_active(obs_vector , report_step));
      
      enkf_tui_plot_RFT__(fs , plot_config , model_config , ensemble_config , obs_vector , obs_key , report_step);
    }
  }
}



void enkf_tui_plot_sensitivity(void * arg) {
  enkf_main_type             * enkf_main       = enkf_main_safe_cast( arg );
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  const plot_config_type    * plot_config      = enkf_main_get_plot_config( enkf_main );
  
  
  enkf_fs_type               * fs              = enkf_main_get_fs(enkf_main);
  const int last_report                        = enkf_main_get_total_length( enkf_main );
  const int ens_size    		       = ensemble_config_get_size(ensemble_config);
  const int prompt_len  		       = 45;                   
  const enkf_config_node_type * config_node_x;
  const enkf_config_node_type * config_node_y;
  double * x 	 = util_malloc( ens_size * sizeof * x , __func__);
  double * y 	 = util_malloc( ens_size * sizeof * y , __func__);
  bool   * valid = util_malloc( ens_size * sizeof * valid , __func__);
  state_enum state_x = BOTH;
  state_enum state_y = BOTH; 
  int report_step_x = 0;
  int report_step_y;
  int iens;
  char * user_key_y;
  char * user_key_x  = NULL;
      

    
  /* Loading the parameter on the x-axis */
  {
    char * key_index_x = NULL;
    util_printf_prompt("Parameter on the x-axis (blank for iens): " , prompt_len , '=' , "=> ");
    user_key_x = util_alloc_stdin_line();
    if (strlen(user_key_x) == 0) {
      user_key_x = util_realloc_string_copy(user_key_x , "Ensemble member");
      config_node_x = NULL;
    } else {
      config_node_x = ensemble_config_user_get_node( ensemble_config , user_key_x , &key_index_x);
      if (config_node_x == NULL) {
	fprintf(stderr,"** Sorry - could not find any nodes with the key:%s \n",user_key_x);
	util_safe_free(key_index_x);
	free(x);
	free(y);
	free(valid);
	free(user_key_x);
	return;
      }
    }

    if (config_node_x == NULL) {
      /* x-axis just contains iens. */
      for (iens = 0; iens < ens_size; iens++) {
	x[iens]     = iens;
	valid[iens] = true;
      }
    } else {
      enkf_node_type * node = enkf_node_alloc( config_node_x );
      for (iens = 0; iens < ens_size; iens++) {
	if (enkf_fs_try_fread_node(fs , node , report_step_x , iens , state_x)) 
	  x[iens] = enkf_node_user_get(node , key_index_x , &valid[iens]);
	else
	  valid[iens] = false;
      }
      enkf_node_free( node );
    }
    util_safe_free(key_index_x);
  }

  /* OK - all the x-data has been loaded. */
  
  {
    char * key_index_y;
    util_printf_prompt("Result on the y-axis: " , prompt_len , '=' , "=> ");
    user_key_y    = util_alloc_stdin_line();
    report_step_y = util_scanf_int_with_limits("Report step: ", prompt_len , 0 , last_report);
    
    {
      config_node_y = ensemble_config_user_get_node( ensemble_config , user_key_y , &key_index_y);
      if (config_node_y == NULL) {
	fprintf(stderr,"** Sorry - could not find any nodes with the key:%s \n",user_key_y);
	util_safe_free(key_index_y);
	free(x);
	free(y);
	free(valid);
	free(user_key_y);
	return;
      }
    }
    {
      enkf_node_type * node = enkf_node_alloc( config_node_y );
      
      for (iens = 0; iens < ens_size; iens++) {
	if (valid[iens]) {
	  if (enkf_fs_try_fread_node(fs , node , report_step_y , iens , state_y)) 
	    y[iens] = enkf_node_user_get(node , key_index_y , &valid[iens]);
	  else
	    valid[iens] = false;
	}
      }
      
      enkf_node_free( node );
    }
    util_safe_free(key_index_y);
  }
  /*****************************************************************/
  /* OK - now we have x[], y[] and valid[] - ready for plotting.   */
  
  {
    int valid_count           = 0;
    char * basename  	      = util_alloc_sprintf("%s-%s" , user_key_x , user_key_y);
    char * plot_file 	      = enkf_tui_plot_alloc_plot_file( plot_config , enkf_fs_get_read_dir(fs), basename );
    plot_type * plot 	      = __plot_alloc( plot_config ,  user_key_x , user_key_y , "Sensitivity plot" , plot_file);
    plot_dataset_type  * data = plot_alloc_new_dataset( plot , NULL , PLOT_XY );
    
    for (iens = 0; iens < ens_size; iens++) {
      if (valid[iens]) {
	plot_dataset_append_point_xy( data , x[iens] , y[iens]);
	valid_count++;
      }
    }
    
    plot_dataset_set_style( data , POINTS);
    plot_set_bottom_padding( plot , 0.05);
    plot_set_top_padding( plot    , 0.05);
    plot_set_left_padding( plot   , 0.05);
    plot_set_right_padding( plot  , 0.05);

    if (valid_count > 0) {
      printf("Plot saved in: %s \n",plot_file);
      __plot_show(plot , plot_config , plot_file); /* Frees the plot - logical ehhh. */
    } else {
      printf("Ehh - no data to plot \n");
      plot_free( plot );
    }
    free(basename);
    free(plot_file);
  }
  

  util_safe_free(user_key_y);
  util_safe_free(user_key_x);
  free(x);
  free(y);
  free(valid);
}





void enkf_tui_plot_menu(void * arg) {
  
  enkf_main_type  * enkf_main  = enkf_main_safe_cast( arg );  
  {
    const plot_config_type * plot_config = enkf_main_get_plot_config( enkf_main );
    const char * plot_path  =  plot_config_get_path( plot_config );
    util_make_path( plot_path );
  }
  
  {
    menu_type * menu;
    {
      char            * title      = util_alloc_sprintf("Plot results [case:%s]" , enkf_fs_get_read_dir(  enkf_main_get_fs( enkf_main ))) ;
      menu = menu_alloc(title , "Back" , "bB");
      free(title);
    }
    menu_add_item(menu , "Ensemble plot"    , "eE"                          , enkf_tui_plot_ensemble    , enkf_main , NULL);
    menu_add_item(menu , "Ensemble plot of ALL summary variables"     , "aA" , enkf_tui_plot_all_summary , enkf_main , NULL);
    menu_add_item(menu , "Ensemble plot of GEN_KW parameter"          , "g"  , enkf_tui_plot_GEN_KW      , enkf_main , NULL);
    menu_add_item(menu , "Ensemble plot of ALL ALL GEN_KW parameters" , "G"  , enkf_tui_plot_all_GEN_KW      , enkf_main , NULL);
    menu_add_item(menu , "Observation plot" , "oO" 			    , enkf_tui_plot_observation , enkf_main , NULL);
    menu_add_item(menu , "RFT depth plot"   , "rR" 			    , enkf_tui_plot_RFT_depth   , enkf_main , NULL);
    menu_add_item(menu , "RFT time plot"    , "tT"                          , enkf_tui_plot_RFT_time    , enkf_main , NULL);
    menu_add_item(menu , "RFT plot of all RFT"  , "fF" 			    , enkf_tui_plot_all_RFT     , enkf_main , NULL);
    menu_add_item(menu , "Sensitivity plot"     , "sS"                      , enkf_tui_plot_sensitivity , enkf_main , NULL); 
    menu_add_item(menu , "Histogram"        , "hH"                          , enkf_tui_plot_histogram   , enkf_main , NULL);
    menu_run(menu);
    menu_free(menu);
  }
}
