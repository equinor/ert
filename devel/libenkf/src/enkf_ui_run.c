#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <util.h>
#include <ctype.h>
#include <menu.h>
#include <enkf_main.h>
#include <enkf_sched.h>
#include <void_arg.h>
#include <enkf_ui_util.h>


void enkf_ui_run(enkf_main_type * enkf_main , enkf_sched_type * enkf_sched , int start_report , state_enum __init_state) {
  bool analyzed_start;
  bool prev_enkf_on;
  bool unlink_run_path = true;
  const int num_nodes            = enkf_sched_get_num_nodes(enkf_sched);
  const int schedule_num_reports = enkf_sched_get_schedule_num_reports(enkf_sched);
  const int start_inode          = enkf_sched_get_node_index(enkf_sched , start_report);
  int inode;
  
  if (__init_state == analyzed)
    analyzed_start = true;
  else
    analyzed_start = false;

  
  prev_enkf_on = analyzed_start;
  for (inode = start_inode; inode < num_nodes; inode++) {
    const enkf_sched_node_type * node = enkf_sched_iget_node(enkf_sched , inode);
    state_enum init_state;
    int 	   init_step;
    int 	   report_step1;
    int 	   report_step2;
    int 	   report_stride;
    int 	   report_step;
    int 	   next_report_step;
    bool enkf_on;
    stringlist_type * forward_model;
    
    enkf_sched_node_get_data(node , &report_step1 , &report_step2 , &report_stride , &enkf_on , &forward_model);
    if (inode == start_inode)
      report_step = start_report;
    else
      report_step = report_step1;
    do {
      next_report_step = util_int_min(schedule_num_reports , util_int_min(report_step + report_stride , report_step2));
      init_step = report_step;
      if (prev_enkf_on)
	init_state = analyzed;
      else
	init_state = forecast;
      
      enkf_main_run(enkf_main , init_step , init_state , report_step , next_report_step , enkf_on , unlink_run_path , forward_model);
      report_step  = next_report_step;
      prev_enkf_on = enkf_on;
    } while (next_report_step < report_step2);
  }
}




void enkf_ui_run_start__(void * _void_arg) {
  void_arg_type   * void_arg   = void_arg_safe_cast(_void_arg);
  enkf_main_type  * enkf_main  = void_arg_get_ptr(void_arg , 0);
  enkf_sched_type * enkf_sched = void_arg_get_ptr(void_arg , 1);

  enkf_ui_run(enkf_main , enkf_sched , 0 , analyzed);
}



void enkf_ui_run_restart__(void * _void_arg) {
  void_arg_type   * void_arg   = void_arg_safe_cast(_void_arg);
  enkf_main_type  * enkf_main  = void_arg_get_ptr(void_arg , 0);
  enkf_sched_type * enkf_sched = void_arg_get_ptr(void_arg , 1);
  const int prompt_len  = 35;
  const int last_report = enkf_sched_get_last_report(enkf_sched);
  int start_report;
  state_enum state;

  start_report = util_scanf_int_with_limits("Report step",prompt_len , 0 , last_report);
  state        = enkf_ui_util_scanf_state("Analyzed/forecast" , prompt_len , false);
  
  enkf_ui_run(enkf_main , enkf_sched , start_report , state);
}


void enkf_ui_run_exp__(void * _void_arg) {
  void_arg_type   * void_arg   = void_arg_safe_cast(_void_arg);
  enkf_main_type  * enkf_main  = void_arg_get_ptr(void_arg , 0);
  enkf_sched_type * enkf_sched = void_arg_get_ptr(void_arg , 1);
  int start_report;

  printf("Start from report : ");
  fscanf(stdin, "%d" , &start_report);
  
}





void enkf_ui_run_menu(void * _arg) {
  void_arg_type   * run_arg    = void_arg_safe_cast(_arg);
  /*
    enkf_main_type  * enkf_main  = void_arg_get_ptr(run_arg , 0);
    enkf_sched_type * enkf_sched = void_arg_get_ptr(run_arg , 1);
  */


  menu_type * menu = menu_alloc("EnKF run menu" , "qQ");
  menu_add_item(menu , "Start EnKF run from beginning"         , "sS" , enkf_ui_run_start__     , run_arg);
  menu_add_item(menu , "Restart EnKF run from arbitrary state" , "rR" , enkf_ui_run_restart__ , run_arg);
  menu_add_item(menu , "Run ensemble experiment"               , "xX" , enkf_ui_run_exp__   , run_arg);
  menu_run(menu);
  menu_free(menu);

}
