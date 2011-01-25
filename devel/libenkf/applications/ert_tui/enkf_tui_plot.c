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
#include <gen_obs.h>
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
#include <member_config.h>
#include <double_vector.h>
#include <ert_tui_const.h>
#include <ecl_rft_file.h>
#include <path_fmt.h>

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
    util_alloc_file_components( file , &plot_path , &basename , NULL);
    
    path = util_alloc_filename( plot_path , basename , NULL);
    arg_pack_append_owned_ptr( arg_pack , path , free);
    
    free( plot_path );
    free( basename );
  } else 
    util_abort("%s: unrecognized driver type: %s \n",__func__ , plot_config_get_driver( plot_config ));
  
  plot = plot_alloc(plot_config_get_driver( plot_config ) , arg_pack , false , plot_config_get_logy( plot_config ));
  
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
    const char * viewer = plot_config_get_viewer( plot_config );
    printf("Plot saved in: %s \n",file);
    if (viewer != NULL)
      util_fork_exec(viewer , 1 , (const char *[1]) { file } , false , NULL , NULL , NULL , NULL , NULL);
  }
  /*
    else: the file does not exist - that might be OK?
  */
}







static void enkf_tui_plot_ensemble__(enkf_main_type * enkf_main , 
                                     const enkf_config_node_type * config_node , 
                                     const char * user_key  ,
                                     const char * key_index ,
                                     int step1 , int step2  , 
                                     bool prediction_mode   ,
                                     int iens1 , int iens2  , 
                                     state_enum plot_state) {
                                     
  
  enkf_fs_type               * fs           = enkf_main_get_fs(enkf_main);
  enkf_obs_type              * enkf_obs     = enkf_main_get_obs( enkf_main );
  const plot_config_type     * plot_config  = enkf_main_get_plot_config( enkf_main );
  
  bool  plot_dates             = true;
  const int errorbar_max_obsnr = plot_config_get_errorbar_max( plot_config );
  const bool add_observations  = true;
  const bool            logy   = plot_config_get_logy( plot_config );
  bool  show_plot              = false;
  char * plot_file             = enkf_tui_plot_alloc_plot_file( plot_config , enkf_fs_get_read_dir(fs), user_key );
  plot_type * plot ;
  enkf_node_type * node;
  msg_type * msg;
  bool_vector_type * has_data = bool_vector_alloc( 0 , false );
  int     iens , step;


  if (plot_dates)
    plot =  __plot_alloc(plot_config , "" , /* y akse */ "" ,user_key,plot_file);
  else
    plot =  __plot_alloc(plot_config , "Simulation time (days) ", /* y akse */ "" ,user_key , plot_file);
  
  node = enkf_node_alloc( config_node );
  {
    char * prompt = util_alloc_sprintf("Loading %s member: " , enkf_config_node_get_key(config_node));
    msg = msg_alloc(prompt, false);
    free(prompt);
  }
  msg_show(msg);

  
  {
    double_vector_type * x      = double_vector_alloc(0,0);
    double_vector_type * y      = double_vector_alloc(0,0);
    for (iens = iens1; iens <= iens2; iens++) {
      char msg_label[32];
      char plot_label[32];
      double_vector_reset( x );
      double_vector_reset( y );
      sprintf(msg_label , "%03d" , iens );
      msg_update( msg , msg_label);
      
      if (prediction_mode)
        step2 = member_config_get_sim_length( enkf_main_iget_member_config( enkf_main , iens ) ) - 1;
      
      for (step = step1; step <= step2; step++) {
        
        double sim_days = member_config_iget_sim_days(enkf_main_iget_member_config( enkf_main , iens ) , step , fs);
        time_t sim_time = member_config_iget_sim_time(enkf_main_iget_member_config( enkf_main , iens ) , step , fs);
        
        /* Forecast block */
        if (plot_state & FORECAST) {
          if (enkf_fs_has_node(fs , config_node , step , iens , FORECAST)) {
            bool   valid;
            double value;




            enkf_fs_fread_node(fs , node , step , iens , FORECAST);
            value = enkf_node_user_get( node , key_index , &valid);
            
            if (valid) {
              if (logy && (value <= 0))
                valid = false;
            }
            
            if (valid) {
              double_vector_append(y , value);
              bool_vector_iset(has_data , step , true);
              if (plot_dates) 
                double_vector_append(x , sim_time );
              else
                double_vector_append(x , sim_days );
            }
          } 
        }

        
        /* Analyzed block */
        if (plot_state & ANALYZED) {
          if (enkf_fs_has_node(fs , config_node , step , iens , ANALYZED)) {
            bool valid;
            double value;
            enkf_fs_fread_node(fs , node , step , iens , ANALYZED);
            value = enkf_node_user_get( node , key_index , &valid);

            if (valid) {
              if (logy && (value <= 0))
                valid = false;
            }

            if (valid) {
              double_vector_append(y , value);
              bool_vector_iset(has_data , step , true);
              if (plot_dates) 
                double_vector_append(x , sim_time );
              else
                double_vector_append(x , sim_days );
            }
          } 
        }
      }
      
      if (double_vector_size( x ) > 0) {
        show_plot = true;
      
        /* This is called once for every realization - that is kind of wasted. */
        if (plot_dates) {
          time_t min_time = ( time_t ) double_vector_get_min( x );
          time_t max_time = ( time_t ) double_vector_get_max( x );
          
          plot_set_default_timefmt( plot , min_time , max_time );
        }
      
        sprintf(plot_label , "mem_%03d" , iens);
        __plot_add_data(plot , plot_label , double_vector_size( x ) , double_vector_get_ptr( x ) , double_vector_get_ptr( y ));
      }
    }
    double_vector_free( x );
    double_vector_free( y );
  }


  /*
    Observe that all the observations are 'flattened'.
  */
  if (add_observations) {
    enkf_impl_type impl_type = enkf_config_node_get_impl_type(config_node);
    if ((impl_type == SUMMARY) || (impl_type == FIELD) || (impl_type == GEN_DATA)) {
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
        double  value , std;
        int report_step = -1;
        do {
          report_step = obs_vector_get_next_active_step( obs_vector , report_step);
          if (report_step != -1) {
            if (bool_vector_safe_iget( has_data , report_step)) {   /* Not plotting an observation if we do not have any simulations at the same time. */
              bool valid;

              /**
                 The user index used when calling the user_get function on the
                 gen_obs data type is different depending on whether is called with a
                 data context user_key (as here) or with a observation context
                 user_key (as when plotting an observation plot). See more
                 documentation of the function gen_obs_user_get_data_index(). 
              */
              
              if (impl_type == GEN_DATA)
                gen_obs_user_get_with_data_index( obs_vector_iget_node( obs_vector , report_step ) , key_index , &value , &std , &valid);
              else
                obs_vector_user_get( obs_vector , key_index , report_step , &value , &std , &valid);

              if (valid) {
                if (logy && ((value - std) <= 0.0))
                  valid = false;
              }


              if (valid) {
                /**
                   Should get sim_time directly from the observation - and not inderctly thrugh the member_config object.
                */
                if (plot_dates)
                  double_vector_append( sim_time  , member_config_iget_sim_time(enkf_main_iget_member_config( enkf_main , iens1 ) , report_step , fs));  
                else
                  double_vector_append( sim_time  , member_config_iget_sim_days(enkf_main_iget_member_config( enkf_main , iens1 ) , report_step , fs));  
                
                double_vector_append( obs_value , value );
                double_vector_append( obs_std   , std );

                obs_size += 1;
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
          /*
            Normal plot with errorbars. Observe that the coordinates
            are (x,y1,y2) and NOT (x,y,std_y).
          */

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
  if (show_plot) {
    __plot_show(plot , plot_config , plot_file); /* Frees the plot - logical ehhh. */
  } else {
    printf("No data to plot \n");
    plot_free(plot);
  }
           
  free(plot_file);
  bool_vector_free( has_data );
}



void enkf_tui_plot_GEN_KW__(enkf_main_type * enkf_main , const enkf_config_node_type * config_node , int step1 , int step2 , int iens1 , int iens2) {
  gen_kw_config_type * gen_kw_config        = enkf_config_node_get_ref( config_node );
  stringlist_type * key_list                = gen_kw_config_alloc_name_list( gen_kw_config );
  
  int ikw;
  for (ikw = 0; ikw < stringlist_get_size( key_list ); ikw++) {
    char * user_key = gen_kw_config_alloc_user_key( gen_kw_config , ikw );
    enkf_tui_plot_ensemble__( enkf_main , config_node , user_key , stringlist_iget( key_list , ikw) , step1 , step2 , false , iens1 , iens2 , ANALYZED );
    free( user_key );
  }
  
  stringlist_free( key_list );
}



void enkf_tui_plot_GEN_KW(void * arg) {
  enkf_main_type             * enkf_main       = enkf_main_safe_cast( arg );
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  {
    const char * prompt  = "Which GEN_KW parameter do you want to plot";
    const enkf_config_node_type * config_node = NULL;
    bool  exit_loop = false;

    do {
      char *node_key;
      util_printf_prompt(prompt , PROMPT_LEN , '=' , "=> ");
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
      const int last_report      = enkf_main_get_history_length( enkf_main );

      enkf_tui_util_scanf_report_steps(last_report , PROMPT_LEN , &step1 , &step2);
      enkf_tui_util_scanf_iens_range("Realizations members to plot(0 - %d)" , enkf_main_get_ensemble_size( enkf_main ) , PROMPT_LEN , &iens1 , &iens2);
      
      enkf_tui_plot_GEN_KW__(enkf_main , config_node , step1 , step2 , iens1 , iens2);
    }
  }
}





                         

void enkf_tui_plot_all_GEN_KW(void * arg) {
  enkf_main_type             * enkf_main       = enkf_main_safe_cast( arg );
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  {
    int iens1 , iens2 , step1 , step2 , ikey;   
    stringlist_type * gen_kw_keys = ensemble_config_alloc_keylist_from_impl_type(ensemble_config , GEN_KW);
    const int last_report         = enkf_main_get_history_length( enkf_main );

    enkf_tui_util_scanf_report_steps(last_report , PROMPT_LEN , &step1 , &step2);
    enkf_tui_util_scanf_iens_range("Realizations members to plot(0 - %d)" , enkf_main_get_ensemble_size( enkf_main ) , PROMPT_LEN , &iens1 , &iens2);
    
    for (ikey = 0; ikey < stringlist_get_size( gen_kw_keys ); ikey++) {
      const char * key = stringlist_iget( gen_kw_keys , ikey);
      enkf_config_node_type * config_node = ensemble_config_get_node( ensemble_config , key ); 
      enkf_tui_plot_GEN_KW__(enkf_main , config_node , step1 , step2 , iens1 , iens2);
    }
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
    const char * prompt  = "What do you want to plot (KEY:INDEX)";
    const enkf_config_node_type * config_node;
    char       * user_key;
    
    
    util_printf_prompt(prompt , PROMPT_LEN , '=' , "=> ");
    user_key = util_alloc_stdin_line();
    if (user_key != NULL) {
      const int ens_size    = enkf_main_get_ensemble_size( enkf_main );
      state_enum plot_state = ANALYZED; /* Compiler shut up */
      char * key_index;
      const int last_report = enkf_main_get_history_length( enkf_main );
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
      report_step = util_scanf_int_with_limits("Report step: ", PROMPT_LEN , 0 , last_report);
      {
        enkf_var_type var_type = enkf_config_node_get_var_type(config_node);
        if ((var_type == DYNAMIC_STATE) || (var_type == DYNAMIC_RESULT)) 
          plot_state = enkf_tui_util_scanf_state("Plot Forecast/Analyzed: [F|A]" , PROMPT_LEN , false);
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
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  {
    const bool prediction_mode = false;
    const char * prompt        = "What do you want to plot (KEY:INDEX)";
    const enkf_config_node_type * config_node;
    char * user_key;
    
    util_printf_prompt(prompt , PROMPT_LEN , '=' , "=> ");
    user_key = util_alloc_stdin_line();
    if (user_key != NULL) {
      state_enum plot_state = ANALYZED; /* Compiler shut up */
      char * key_index;
      const int last_report = enkf_main_get_history_length( enkf_main );
      int iens1 , iens2 , step1 , step2;   
            
      config_node = ensemble_config_user_get_node( ensemble_config , user_key , &key_index);
      if (config_node == NULL) {
        fprintf(stderr,"** Sorry - could not find any nodes with the key:%s \n",user_key);
        util_safe_free(key_index);
        return;
      }

      enkf_tui_util_scanf_report_steps(last_report , PROMPT_LEN , &step1 , &step2);
      enkf_tui_util_scanf_iens_range("Realizations members to plot(0 - %d)" , enkf_main_get_ensemble_size( enkf_main ) , PROMPT_LEN , &iens1 , &iens2);
      
      {
        enkf_var_type var_type = enkf_config_node_get_var_type(config_node);
        if ((var_type == DYNAMIC_STATE) || (var_type == DYNAMIC_RESULT)) 
          plot_state = enkf_tui_util_scanf_state("Plot Forecast/Analyzed/Both: [F|A|B]" , PROMPT_LEN , true);
        else if (var_type == PARAMETER)
          plot_state = ANALYZED;
        else
          util_abort("%s: can not plot this type \n",__func__);
      }
      enkf_tui_plot_ensemble__(enkf_main , 
                               config_node , 
                               user_key , 
                               key_index , 
                               step1 , 
                               step2 , 
                               prediction_mode , 
                               iens1 , 
                               iens2 , 
                               plot_state);
      util_safe_free(key_index);
    }
    util_safe_free( user_key );
  }
}
        
           

void enkf_tui_plot_all_summary(void * arg) {
  enkf_main_type             * enkf_main       = enkf_main_safe_cast( arg );
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  int iens1 , iens2 , step1 , step2;   
  bool prediction_mode;


  
  {
    bool default_used;
    step1 = enkf_tui_util_scanf_int_with_default( "Starting plotting at report step [default: 0]"      , PROMPT_LEN , &default_used);
    if (default_used)
      step1 = 0;
    step2 = enkf_tui_util_scanf_int_with_default( "Stop plotting at report step [default: everything]" , PROMPT_LEN , &prediction_mode);
  }
  enkf_tui_util_scanf_iens_range("Realizations members to plot(0 - %d) [default: all]" , enkf_main_get_ensemble_size( enkf_main ) , PROMPT_LEN , &iens1 , &iens2);
  
  {
    stringlist_type * summary_keys = ensemble_config_alloc_keylist_from_impl_type(ensemble_config , SUMMARY);
    int ikey;
    for (ikey = 0; ikey < stringlist_get_size( summary_keys ); ikey++) {
      const char * key = stringlist_iget( summary_keys , ikey);
      
      enkf_tui_plot_ensemble__(enkf_main,
                               ensemble_config_get_node( ensemble_config , key ),
                               key , 
                               NULL , 
                               step1 , step2   , 
                               prediction_mode ,
                               iens1 , iens2   , 
                               BOTH);
                               
      
    }
    stringlist_free( summary_keys );
  }
}


          


void enkf_tui_plot_observation(void * arg) {
  enkf_main_type             * enkf_main       = enkf_main_safe_cast( arg );
  enkf_obs_type              * enkf_obs        = enkf_main_get_obs( enkf_main );
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  const plot_config_type     * plot_config     = enkf_main_get_plot_config( enkf_main );
  {
    const int ens_size = enkf_main_get_ensemble_size( enkf_main );
    const char * prompt  = "What do you want to plot (KEY:INDEX)";
    enkf_fs_type   * fs   = enkf_main_get_fs(enkf_main);
    const obs_vector_type * obs_vector;
    char * user_key;
    char * index_key;

    util_printf_prompt(prompt , PROMPT_LEN , '=' , "=> ");
    user_key = util_alloc_stdin_line();
    if (user_key != NULL) {
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
            report_step = enkf_tui_util_scanf_report_step(enkf_main_get_history_length( enkf_main ) , "Report step" , PROMPT_LEN);
        } while (!obs_vector_iget_active(obs_vector , report_step));
        {
          enkf_node_type * enkf_node = enkf_node_alloc( config_node );
          msg_type * msg = msg_alloc("Loading realization: ",false);
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
          {
            double sum1 = 0;
            double sum2 = 0;
            int    num  = 0;
            
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
                if (valid) {
                  plot_dataset_append_point_xy( forecast_data , iens , y);
                  sum1 += y;
                  sum2 += y*y;
                  num  += 1;
                }
              }
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
    util_safe_free( user_key );
  }
}


void enkf_tui_plot_RFT__(enkf_main_type * enkf_main,
                         const char * obs_key , 
                         int report_step) {
  
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config( enkf_main );
  const plot_config_type     * plot_config     = enkf_main_get_plot_config( enkf_main );
  enkf_obs_type              * enkf_obs        = enkf_main_get_obs( enkf_main );
  enkf_fs_type               * fs              = enkf_main_get_fs( enkf_main );
  const obs_vector_type      * obs_vector      = enkf_obs_get_vector( enkf_obs , obs_key );
  const char                 * state_kw        = obs_vector_get_state_kw( obs_vector );
  plot_type                  * plot;
  enkf_node_type             * node;
  
  const int ens_size                  = enkf_main_get_ensemble_size( enkf_main );
  enkf_config_node_type * config_node = ensemble_config_get_node( ensemble_config , state_kw );
  field_config_type * field_config    = enkf_config_node_get_ref( config_node );
  field_obs_type    * field_obs       = obs_vector_iget_node( obs_vector , report_step );
  char * plot_file;
  
  plot_file = enkf_tui_plot_alloc_plot_file(plot_config , enkf_fs_get_read_dir(fs), obs_key );
  plot = __plot_alloc(plot_config , state_kw , "Depth" , obs_key , plot_file);
  {
    msg_type * msg             = msg_alloc("Loading realization: ",false);
    const int * i              = field_obs_get_i(field_obs);
    const int * j              = field_obs_get_j(field_obs);
    const int * k              = field_obs_get_k(field_obs);
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
      ecl_grid_get_xyz3(grid , i[l] , j[l] , k[l] , &xpos , &ypos , &zpos);
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
    const char * prompt  = "Which RFT observation: ";

    const obs_vector_type * obs_vector = NULL;
    char  *obs_key;
    int    report_step;
    while (true) {
      util_printf_prompt(prompt , PROMPT_LEN , '=' , "=> ");
      obs_key = util_alloc_stdin_line( );
      if (obs_key != NULL) {
        if (enkf_obs_has_key(enkf_obs , obs_key)) {
          obs_vector = enkf_obs_get_vector( enkf_obs , obs_key );
          if (obs_vector_get_impl_type( obs_vector ) == FIELD_OBS)
            break; /* Jumping out with a valid obs_vector pointer. */
          else {
            fprintf(stderr,"Observation key:%s does not correspond to a field observation.\n",obs_key);
            obs_vector = NULL;
          }
        } else
          fprintf(stderr,"Do not have observation key:%s \n",obs_key);
      } else
        break; /* Jumping out on blank input */
    }

    if (obs_vector != NULL) {
      do {
        if (obs_vector_get_num_active( obs_vector ) == 1)
          report_step = obs_vector_get_active_report_step( obs_vector );
        else
          report_step = enkf_tui_util_scanf_report_step(enkf_main_get_history_length( enkf_main ) , "Report step" , PROMPT_LEN);
      } while (!obs_vector_iget_active(obs_vector , report_step));
      *_obs_key = obs_key;
      *_report_step = report_step;
    }
  }
}



void enkf_tui_plot_RFT_depth(void * arg) {
  enkf_main_type             * enkf_main       = enkf_main_safe_cast( arg );
  enkf_obs_type              * enkf_obs        = enkf_main_get_obs( enkf_main );
  {
    char * obs_key;
    int report_step;
    obs_vector_type * obs_vector;
    
    enkf_tui_plot_select_RFT(enkf_main , &obs_key , &report_step);
    obs_vector = enkf_obs_get_vector( enkf_obs , obs_key );

    enkf_tui_plot_RFT__(enkf_main , obs_key , report_step);
    free( obs_key );
  }
}



void enkf_tui_plot_RFT_time(void * arg) {
  enkf_main_type             * enkf_main       = enkf_main_safe_cast( arg );
  enkf_obs_type              * enkf_obs        = enkf_main_get_obs( enkf_main );
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
    step2      = enkf_main_get_history_length( enkf_main );
    iens1      = 0;
    iens2      = enkf_main_get_ensemble_size( enkf_main ) - 1;
    plot_state = BOTH;
    state_kw   = enkf_config_node_get_key( config_node );
    {
      int block_nr,i,j,k;
      const field_obs_type * field_obs = obs_vector_iget_node( obs_vector , report_step );
      for (block_nr = 0; block_nr < field_obs_get_size( field_obs ); block_nr++) {
        field_obs_iget_ijk( field_obs , block_nr , &i , &j , &k);
        index_key = util_realloc_sprintf( index_key , "%d,%d,%d"    , i+1,j+1,k+1);
        user_key  = util_realloc_sprintf( user_key  , "%s:%d,%d,%d" , state_kw , i+1,j+1,k+1);
        enkf_tui_plot_ensemble__(enkf_main , config_node , user_key , index_key , step1 , step2 , false , iens1 , iens2 , plot_state);
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
  {
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
          report_step = enkf_tui_util_scanf_report_step(enkf_main_get_history_length( enkf_main ) , "Report step" , PROMPT_LEN);
      } while (!obs_vector_iget_active(obs_vector , report_step));
      
      enkf_tui_plot_RFT__(enkf_main , obs_key , report_step);
    }
  }
}


int enkf_tui_plot_read_rft_obs(enkf_main_type * enkf_main, 
			  char * wellname, 
			  double_vector_type * UTM_x, 
			  double_vector_type * UTM_y, 
			  double_vector_type * MD, 
			  double_vector_type * TVD_z, 
			  double_vector_type * RFT_obs){
  const model_config_type * model_config = enkf_main_get_model_config( enkf_main ); 
  const char * pathname = model_config_get_rftpath( model_config );
  path_fmt_type * pathname_fmt = path_fmt_alloc_directory_fmt(pathname);
  strcat(wellname, ".txt"); /* This will append .txt to the wellname*/
  char * pathandfilename = path_fmt_alloc_file(pathname_fmt, false, wellname);
  FILE * stream = util_fopen( pathandfilename , "r" ); 
  int lines = util_count_content_file_lines( stream );
  rewind(stream);
  double utm_x, utm_y, md, tvd_z, rft_obs;
  bool at_eof;
  char ** token_list;
  int tokens;
  for ( int i = 0; i < lines; i++ ) {
    char * line = util_fscanf_alloc_line( stream , &at_eof );
    util_split_string(line , " \t" , &tokens , &token_list);
    if( tokens == 5 ){
      if ( util_sscanf_double( token_list[0] , &utm_x ) && util_sscanf_double( token_list[1] , &utm_y ) && util_sscanf_double( token_list[2] , &md ) && util_sscanf_double( token_list[3] , &tvd_z ) && util_sscanf_double( token_list[4] , &rft_obs )){
	double_vector_iset( UTM_x  , i, utm_x );
	double_vector_iset( UTM_y  , i, utm_y );
	double_vector_iset( MD     , i, md );
	double_vector_iset( TVD_z  , i, tvd_z);
	double_vector_iset( RFT_obs, i, rft_obs);
      }
      else{
	util_abort("%s: RFT file has to be on the format UTM_X; UTM_Y; MD; TVD_Z; RFT \n",__func__ , pathandfilename);
      }
    }
    else{
      util_abort("%s: RFT file has to be on the format UTM_X; UTM_Y; MD; TVD_Z; RFT \n",__func__ , pathandfilename);
    }
    free( line );
  }
  free(pathandfilename);
  path_fmt_free(pathname_fmt);
  util_free_stringlist(token_list, tokens);
  wellname=strtok(wellname,".");/* This will remove the appended .txt from the wellname*/
  fclose(stream);
  return lines;
}



void enkf_tui_plot_RFTS__(enkf_main_type * enkf_main , 
				     const char * wellname   ,
				     double_vector_type * MD, 
				     double_vector_type * RFT_obs,
				     double_vector_type * RFT_refcase,
			             bool_vector_type * refcase_has_data,
                                     vector_type * pressure_container, 
				     int_vector_type * active,
			             bool rft_file_exists,
			             vector_type * has_data_container) {
                                     
  const int ens_size                        = enkf_main_get_ensemble_size( enkf_main );
  enkf_fs_type               * fs           = enkf_main_get_fs(enkf_main);
  const plot_config_type     * plot_config  = enkf_main_get_plot_config( enkf_main );
  bool  show_plot              = false;
  char * plot_file             = enkf_tui_plot_alloc_plot_file( plot_config , enkf_fs_get_read_dir(fs), wellname );
  plot_type * plot ;
  plot =  __plot_alloc(plot_config , "RFT pressure", "MD" , wellname, plot_file);
  {
    show_plot = true;
  }
  plot_dataset_type * rft_obs_value     = plot_alloc_new_dataset( plot , "observation"       , PLOT_XY );
  plot_dataset_type * rft_refcase_value = plot_alloc_new_dataset( plot , "refcase"       , PLOT_XY );

  plot_dataset_set_style( rft_obs_value , POINTS );
  plot_dataset_set_style( rft_refcase_value , POINTS );
          
  plot_dataset_set_point_color( rft_obs_value , 15);
  plot_dataset_set_point_color( rft_refcase_value , 15);

  plot_dataset_set_symbol_type( rft_obs_value , 5);
  plot_dataset_set_symbol_type( rft_refcase_value , 17);
  
  
  for (int nobs = 0; nobs < double_vector_size(RFT_obs); nobs++){
    double rft_obs_numeric     = double_vector_iget(RFT_obs , nobs);
    double md_numeric          = double_vector_iget(MD , nobs);
    plot_dataset_append_point_xy( rft_obs_value, rft_obs_numeric , md_numeric);
    if( bool_vector_iget(refcase_has_data, nobs)){
      double rft_refcase_numeric = double_vector_iget(RFT_refcase , nobs);
      plot_dataset_append_point_xy( rft_refcase_value, rft_refcase_numeric , md_numeric);
    }
  }
  
  /*
    Now that the refcase and observations are handeled, simulated rfts are to be plotted.
  */ 
  if(rft_file_exists){
    for (int iens=0; iens < ens_size; iens++){
      const double_vector_type * simulated_pressure = vector_iget_const(pressure_container, iens);
      const bool_vector_type * has_data = vector_iget_const(has_data_container, iens);
      for (int nobs = 0; nobs < double_vector_size(RFT_obs); nobs++){
	if (bool_vector_iget(has_data, nobs)){
	  plot_dataset_type * iplot  = plot_alloc_new_dataset( plot , NULL , PLOT_XY );
	  double rft_sim_numeric     = double_vector_iget(simulated_pressure , nobs);
	  double md_numeric          = double_vector_iget(MD , nobs);
	  plot_dataset_append_point_xy( iplot, rft_sim_numeric , md_numeric);
	  plot_dataset_set_style( iplot , POINTS );
	  plot_dataset_set_point_color( iplot , (iens % 13)+1); /*Can choose between 16 colors, but we dont want 0 which is white or reserved 14 and 15*/ 
	}
      }
    }
  }
  plot_invert_y_axis(plot);
  plot_set_bottom_padding( plot , 0.05);
  plot_set_top_padding( plot    , 0.05);
  plot_set_left_padding( plot   , 0.05);
  plot_set_right_padding( plot  , 0.05);
  
  if (show_plot) {
    __plot_show(plot , plot_config , plot_file); /* Frees the plot - logical ehhh. */
  } else {
    printf("No data to plot \n");
    plot_free(plot);
  }
  
  free(plot_file);
}



void enkf_tui_plot_RFT_simIn(enkf_main_type * enkf_main, path_fmt_type * runpathformat, const path_fmt_type * caseformat, char * wellname , time_t recording_time){
  const int ens_size    = enkf_main_get_ensemble_size( enkf_main );
  /*
    Start by reading RFT measurment
  */
  double_vector_type * UTM_x   = double_vector_alloc( 0 , 0); /*At this stage unknown size, use 0*/
  double_vector_type * UTM_y   = double_vector_alloc( 0 , 0); /*At this stage unknown size, use 0*/
  double_vector_type * MD      = double_vector_alloc( 0 , 0); /*At this stage unknown size, use 0*/
  double_vector_type * TVD_z   = double_vector_alloc( 0 , 0); /*At this stage unknown size, use 0*/
  double_vector_type * RFT_obs = double_vector_alloc( 0 , 0); /*At this stage unknown size, use 0*/
  int lines = enkf_tui_plot_read_rft_obs(enkf_main, wellname, UTM_x, UTM_y, MD, TVD_z, RFT_obs);
  /*
    Find ijk-list
  */
  char * caseending = path_fmt_alloc_path(caseformat, false, 0); //Use the grid in ensmember 0
  char * casename = path_fmt_alloc_file(runpathformat , false, 0, caseending);//Use the grid in ensmember 0
  ecl_grid_type * grid = ecl_grid_load_case( casename );
  int_vector_type * i_values = int_vector_alloc( lines , 0 );
  int_vector_type * j_values = int_vector_alloc( lines , 0 );
  int_vector_type * k_values = int_vector_alloc( lines , 0 );
  int_vector_type * active   = int_vector_alloc( lines , 0 );
  int start_index = 0;
  for (int nobs =0; nobs<lines; nobs++){
    int start_index = 0;
    int i; int j; int k;
    int global_index = ecl_grid_get_global_index_from_xyz(grid,double_vector_iget(UTM_x,nobs) ,double_vector_iget(UTM_y,nobs) ,double_vector_iget(TVD_z,nobs) ,start_index);
    ecl_grid_get_ijk1(grid , global_index, &i, &j , &k);
    int is_active = ecl_grid_get_active_index1(grid , global_index);
    int_vector_iset(i_values, nobs, i);
    int_vector_iset(j_values, nobs, j);
    int_vector_iset(k_values, nobs, k);
    int_vector_iset(active  , nobs, is_active);
    start_index = global_index;
  }
  ecl_grid_free(grid);
  /*
    Find refcase rfts
  */
  double_vector_type * RFT_refcase = double_vector_alloc( 0 , 0);
  bool_vector_type * refcase_has_data = bool_vector_alloc(0, false);
  const char * refcase_name = ecl_config_get_refcase_name( enkf_main_get_ecl_config(enkf_main));
  const char * refcase_file_name = ecl_rft_file_alloc_case_filename(refcase_name );
  if (refcase_file_name == NULL){
    util_abort("%s: Cannot find eclipse RFT file",__func__ , refcase_file_name);
  }
  ecl_rft_file_type * rft_refcase_file = ecl_rft_file_alloc( refcase_file_name );
  if (refcase_file_name == NULL){
    util_abort("%s: Cannot find eclipse RFT file",__func__ , refcase_file_name);
  }
  const ecl_rft_node_type * rft_refcase_node = ecl_rft_file_get_well_time_rft( rft_refcase_file , wellname , recording_time);  
  if(rft_refcase_node == NULL){
    printf("No RFT information exists for %s in refcase.\n", wellname);
  }
  else{
    for( int nobs = 0; nobs < lines; nobs++){
      if( int_vector_iget(active,nobs) > -1){
	int cell_index = ecl_rft_node_lookup_ijk( rft_refcase_node , int_vector_iget(i_values,nobs), int_vector_iget(j_values,nobs),int_vector_iget(k_values,nobs) ); //lookup cell
	if(cell_index > -1){
	  double pressure_value = ecl_rft_node_iget_pressure( rft_refcase_node , cell_index); // Pressure
	  double_vector_append(RFT_refcase, pressure_value);
	  bool_vector_append(refcase_has_data, true);
	}
	else{
	  double_vector_append(RFT_refcase, 0.0);
	  bool_vector_append(refcase_has_data, false);
	}
      }
      else {
	double_vector_append(RFT_refcase, 0.0);
	bool_vector_append(refcase_has_data, false);
      }
    }
  }
  ecl_rft_file_free(rft_refcase_file);
  /*
    Get the simulated RFTs
  */
  vector_type * pressure_container = vector_alloc_new();
  vector_type * has_data_container = vector_alloc_new();
  char * caseending1 = path_fmt_alloc_path(caseformat, false, 0);
  char * casename1 = path_fmt_alloc_file(runpathformat , false, 0, caseending1);
  const char * case_file_name1 = ecl_rft_file_alloc_case_filename(casename1 );
  bool eclipse_rft_exists = false;
  if (case_file_name1 == NULL){
    util_abort("%s: Cannot find eclipse RFT file",__func__ , case_file_name1);
  }
  else{
    eclipse_rft_exists = true;
    for (int iens = 0; iens<ens_size; iens++){
      double_vector_type * simulated_pressures = double_vector_alloc(lines, 0.0);
      bool_vector_type * has_data = bool_vector_alloc(lines, true);
      char * caseending = path_fmt_alloc_path(caseformat, false, iens);
      char * casename = path_fmt_alloc_file(runpathformat , false, iens, caseending);
      const char * case_file_name = ecl_rft_file_alloc_case_filename(casename );
      ecl_rft_file_type * rftfile = ecl_rft_file_alloc( case_file_name );
      const ecl_rft_node_type * rftnode = ecl_rft_file_get_well_time_rft( rftfile , wellname , recording_time);
      if(rftnode == NULL){
	printf("No RFT information exists for %s:\n", wellname);
      }
      else{
	for( int nobs = 0; nobs < lines; nobs++){
	  if( int_vector_iget(active,nobs) > -1){
	    int cell_index = ecl_rft_node_lookup_ijk( rftnode , int_vector_iget(i_values,nobs), int_vector_iget(j_values,nobs),int_vector_iget(k_values,nobs) ); //lookup cell
	    double pressure_value = ecl_rft_node_iget_pressure( rftnode , cell_index); // Pressure
	    double_vector_iset(simulated_pressures,nobs , pressure_value);
	    if(cell_index > -1)
	      bool_vector_iset(has_data, nobs, true);
	    else
	      bool_vector_iset(has_data, nobs, false);
	  }
	  else {
	    double_vector_iset(simulated_pressures,nobs ,0.0);
	    bool_vector_iset(has_data, nobs, false);
	  }
	}
      }
      ecl_rft_file_free(rftfile);
      vector_append_owned_ref( pressure_container , simulated_pressures , double_vector_free__ );
      vector_append_owned_ref( has_data_container , has_data , bool_vector_free__ );
    }
  }
  /*
    Do the actual plotting
  */
  enkf_tui_plot_RFTS__( enkf_main , wellname , MD, RFT_obs, RFT_refcase, refcase_has_data, pressure_container, active, eclipse_rft_exists, has_data_container);
  double_vector_free( UTM_x );
  double_vector_free( UTM_y );
  double_vector_free( MD  );
  double_vector_free( TVD_z );
  double_vector_free( RFT_obs );
  double_vector_free( RFT_refcase );
  bool_vector_free( refcase_has_data );
  vector_free( pressure_container );  
  vector_free( has_data_container );
  free( caseending );
  free( caseending1 );
  free( casename );
  free( casename1 );
  int_vector_free( i_values );
  int_vector_free( j_values );
  int_vector_free( k_values );
  int_vector_free( active );
};


int enkf_tui_plot_read_rft_config(const char * rft_config_file, stringlist_type * wellnames, time_t_vector_type * dates){
  int lines = 0;
  int day, month, year;
  if ( rft_config_file != NULL ){
    printf( "Reading RFT wellnames and dates \n" );
    FILE * stream = util_fopen( rft_config_file , "r" );
    if(stream == NULL)
      util_abort("%s: RFT config file is NULL \n",__func__ , rft_config_file);  
    
    lines = util_count_content_file_lines( stream );
    rewind(stream);
    bool at_eof;
    char ** token_list;
    int tokens;
    for ( int i = 0; i < lines; i++ ) {
      char * line = util_fscanf_alloc_line( stream , &at_eof );
      util_split_string(line , " \t" , &tokens , &token_list);
      char * name = token_list[0];
      char * ownname = util_alloc_string_copy(name);
      if( tokens == 4 ){
	stringlist_append_owned_ref( wellnames , ownname );
	if ( util_sscanf_int( token_list[1] , &day ) && util_sscanf_int( token_list[2] , &month ) && util_sscanf_int( token_list[3] , &year ) ){
	  time_t recording_time = util_make_date(day , month , year);
	  time_t_vector_append(dates, recording_time);
	}
	else{
	  util_abort("%s: RFT config file has to be on the format NAME DAY MONTH YEAR \n",__func__ , rft_config_file);
	}
      }
      else{
	util_abort("%s: RFT config file has to be on the format NAME DAY MONTH YEAR \n",__func__ , rft_config_file);
      }
      free( line );
      free( name );
    }
    fclose(stream);  
  }
  else{
    printf("RFT config file is not specified.");
  }
  return lines;
}

void enkf_tui_plot_RFT_sim_all( void * arg) {
  enkf_main_type * enkf_main = enkf_main_safe_cast( arg );
  const model_config_type * model_config = enkf_main_get_model_config( enkf_main );
  const char * rft_config_file = enkf_main_get_rft_config_file( enkf_main );
  stringlist_type * wellnames = stringlist_alloc_new();
  time_t_vector_type * dates = time_t_vector_alloc(0,0);
  int lines = enkf_tui_plot_read_rft_config(rft_config_file, wellnames, dates);
  path_fmt_type * runpathformat = model_config_get_runpath_fmt( model_config );
  const path_fmt_type * caseformat = ecl_config_get_eclbase_fmt(enkf_main_get_ecl_config(enkf_main));
  for (int i = 0; i<lines; i++){
    char * wellname = stringlist_iget_copy(wellnames, i);
    time_t  recording_time = time_t_vector_iget(dates, i);
    enkf_tui_plot_RFT_simIn(enkf_main, runpathformat, caseformat, wellname, recording_time);
  }
  stringlist_free(wellnames);
  time_t_vector_free(dates);
}









void enkf_tui_plot_sensitivity(void * arg) {
  enkf_main_type             * enkf_main       = enkf_main_safe_cast( arg );
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  const plot_config_type    * plot_config      = enkf_main_get_plot_config( enkf_main );
  
  
  enkf_fs_type               * fs              = enkf_main_get_fs(enkf_main);
  const int last_report                        = enkf_main_get_history_length( enkf_main );
  const int ens_size                           = enkf_main_get_ensemble_size( enkf_main );
  const enkf_config_node_type * config_node_x;
  const enkf_config_node_type * config_node_y;
  double * x     = util_malloc( ens_size * sizeof * x , __func__);
  double * y     = util_malloc( ens_size * sizeof * y , __func__);
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
    util_printf_prompt("Parameter on the x-axis (blank for iens): " , PROMPT_LEN , '=' , "=> ");
    user_key_x = util_alloc_stdin_line();
    if (user_key_x == NULL) {
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
    util_printf_prompt("Result on the y-axis: " , PROMPT_LEN , '=' , "=> ");
    user_key_y    = util_alloc_stdin_line();
    report_step_y = util_scanf_int_with_limits("Report step: ", PROMPT_LEN , 0 , last_report);
    
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
    char * basename           = util_alloc_sprintf("%s-%s" , user_key_x , user_key_y);
    char * plot_file          = enkf_tui_plot_alloc_plot_file( plot_config , enkf_fs_get_read_dir(fs), basename );
    plot_type * plot          = __plot_alloc( plot_config ,  user_key_x , user_key_y , "Sensitivity plot" , plot_file);
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



static void enkf_tui_toggle_logy(void * arg) {
  arg_pack_type * arg_pack       = arg_pack_safe_cast( arg );
  plot_config_type * plot_config = arg_pack_iget_ptr( arg_pack , 0 );
  menu_item_type * menu_item     = arg_pack_iget_ptr( arg_pack , 1 );

  plot_config_toggle_logy( plot_config );
  if (plot_config_get_logy( plot_config ))
    menu_item_set_label(menu_item , "Use normal Y-axis");
  else
    menu_item_set_label(menu_item , "Use logarithmic Y-axis");
}


static void plot_test( void * arg ) {
  const int length = 100;
  const double xmin = 0;
  const double xmax = 3.14159265 * 2.0;

  double x[length];
  double y[length];
  const int num_plot = 2048;
  enkf_main_type             * enkf_main       = enkf_main_safe_cast( arg );
  const plot_config_type    * plot_config      = enkf_main_get_plot_config( enkf_main );
  plot_config_set_viewer( plot_config , NULL );
  for (int plot_nr = 0; plot_nr < num_plot; plot_nr++) {
    char * filename = util_alloc_sprintf("/tmp/plot_%d.png" , plot_nr);
    plot_type * plot = __plot_alloc(plot_config , "X" , "Y" , "Tittel" , filename );
    
    for (int ix=0; ix < length; ix++) {
      x[ix] = xmin + ix*(xmax - xmin) / ( length - 1);
      y[ix] = cos( x[ix] );

      __plot_add_data( plot , NULL , length , x , y );
    }
    
    __plot_show( plot , plot_config , filename );
    free( filename );
  }
}



void enkf_tui_plot_menu(void * arg) {
  
  enkf_main_type  * enkf_main  = enkf_main_safe_cast( arg );  
  plot_config_type * plot_config = enkf_main_get_plot_config( enkf_main );
  {
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
    menu_add_item(menu , "Ensemble plot"    , "eE"                           , enkf_tui_plot_ensemble    , enkf_main , NULL);
    menu_add_item(menu , "Ensemble plot of ALL summary variables"     , "aA" , enkf_tui_plot_all_summary , enkf_main , NULL);
    menu_add_item(menu , "Ensemble plot of GEN_KW parameter"          , "g"  , enkf_tui_plot_GEN_KW      , enkf_main , NULL);
    menu_add_item(menu , "Ensemble plot of ALL ALL GEN_KW parameters" , "G"  , enkf_tui_plot_all_GEN_KW  , enkf_main , NULL);
    menu_add_item(menu , "Observation plot" , "oO"                           , enkf_tui_plot_observation , enkf_main , NULL);
    menu_add_item(menu , "RFT depth plot"   , "rR"                           , enkf_tui_plot_RFT_depth   , enkf_main , NULL);
    menu_add_item(menu , "RFT time plot"    , "tT"                           , enkf_tui_plot_RFT_time    , enkf_main , NULL);
    menu_add_item(menu , "RFT plot of all RFT"  , "fF"                       , enkf_tui_plot_all_RFT     , enkf_main , NULL);
    menu_add_item(menu , "Plot RFT and simulated pressure vs. MD" , "iI"     , enkf_tui_plot_RFT_sim_all , enkf_main , NULL);
    menu_add_item(menu , "Sensitivity plot"     , "sS"                       , enkf_tui_plot_sensitivity , enkf_main , NULL); 
    menu_add_item(menu , "Histogram"        , "hH"                           , enkf_tui_plot_histogram   , enkf_main , NULL);
    menu_add_separator(menu);
    {
      menu_item_type * menu_item;
      arg_pack_type * arg_pack = arg_pack_alloc();
      menu_item = menu_add_item(menu , "" , "lL" , enkf_tui_toggle_logy , arg_pack , arg_pack_free__);
      arg_pack_append_ptr( arg_pack , plot_config );
      arg_pack_append_ptr( arg_pack , menu_item );
      plot_config_toggle_logy( plot_config );
      enkf_tui_toggle_logy( arg_pack );   /* This sets the label */
    }
    menu_add_item( menu , "Test" , "xX" , plot_test , enkf_main , NULL);
    menu_run(menu);
    menu_free(menu);
  }
}
