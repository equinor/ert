/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'havana_fault_config.c' is part of ERT - Ensemble based Reservoir Tool. 
    
   ERT is free software: you can redistribute it and/or modify 
   it under the terms of the GNU General Public License as published by 
   the Free Software Foundation, either version 3 of the License, or 
   (at your option) any later version. 
    
   ERT is distributed in the hope that it will be useful, but WITHOUT ANY 
   WARRANTY; without even the implied warranty of MERCHANTABILITY or 
   FITNESS FOR A PARTICULAR PURPOSE.   
    
   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html> 
   for more details. 
*/

#include <stdlib.h>
#include <string.h>
#include <util.h>
#include <gen_kw_config.h>
#include <enkf_util.h>
#include <enkf_macros.h>
#include <assert.h>
#include <subst.h>
#include <havana_fault_config.h>



/** 
    A purely local type
*/

struct fault_group_struct {
  int     size;
  char  **fault_names;
  char   *modify_template;
  char   *group_name;
};


#define HAVANA_FAULT_CONFIG_TYPE_ID 887412


struct havana_fault_config_struct 
{
  int                   __type_id;
  gen_kw_config_type  * gen_kw_config;
  char 	    	      * havana_executable;
  char 	    	      * unfaulted_GRDECL_file;
  char                * input_fault_path;
  char                * update_template;
  int                   num_fault_groups;
  fault_group_type   ** fault_groups;
};



/*****************************************************************/
/** 
   Implementation of fault_group_type follows here. This is just
   convenience class for handling groups of faults. No functionality
   of this type should be exported.
*/
   

static fault_group_type * fault_group_alloc(const char * group_name , int size, const char ** fault_names , const char * modify_template) {
  fault_group_type * group = util_malloc(sizeof * group , __func__);
  group->size              = size;
  group->modify_template   = util_alloc_string_copy(modify_template);
  group->fault_names       = util_alloc_stringlist_copy(fault_names , size);
  group->group_name        = util_alloc_string_copy(group_name);
  return group;
}


static void fault_group_free(fault_group_type * group) {
  free(group->modify_template);
  free(group->group_name);
  util_free_stringlist(group->fault_names , group->size);
  free(group);
}


static void fault_group_faultlist_append(const fault_group_type * group , FILE * stream) {
  int i;
  for (i=0; i < group->size; i++)
    fprintf(stream , "%s\n",group->fault_names[i]);
}

/**
  This function adds symbolic links from the path tmp_PFM_path to the
  real fault files in PFM_path. This is done so that several instances
  of havana can access the same (original) faults at the same
  time. The '.faultlist' file is located in the tmp_PFM_path.

  Observe that the argument PFM_path in the xxx_fprintf_faultlist()
  function should in general correspond to the value of the
  tmp_PFM_path variable.
*/

static void fault_group_link_faults(const fault_group_type * group , const char * PFM_path , const char * tmp_PFM_path) {
  int i;
  for (i =  0; i < group->size; i++) {
    char * target = util_alloc_filename(PFM_path     , group->fault_names[i] , NULL);
    char * link   = util_alloc_filename(tmp_PFM_path , group->fault_names[i] , NULL);
    util_make_slink(target , link);
    free(target);
    free(link);
  }
}


static void fault_group_fprintf_faultlist(const fault_group_type * group , const char * PFM_path) {
  char * filename = util_alloc_filename(PFM_path , ".faultlist" , NULL);
  FILE * stream   = util_fopen(filename , "w");
  fprintf(stream , "%d\n",group->size);
  fault_group_faultlist_append(group , stream);
  fclose(stream);
  free(filename);
}


static  void fault_group_fprintf_ALL_faultlist(const fault_group_type **group_list , int num_fault_groups , const char * PFM_path){
  int total_faults = 0;
  int igroup;
  for (igroup = 0; igroup < num_fault_groups; igroup++) 
    total_faults += group_list[igroup]->size;
  {
    char * filename = util_alloc_filename(PFM_path , ".faultlist" , NULL);
    FILE * stream = util_fopen(filename , "w");
    fprintf(stream , "%d\n", total_faults);
    
    for (igroup = 0; igroup < num_fault_groups; igroup++) 
      fault_group_faultlist_append(group_list[igroup] , stream);

    fclose(stream);
    free(filename);
  }
}



static void fault_group_run_havana(const fault_group_type * group , const subst_list_type * subst_list , const char * run_path , const char * PFM_path , const char * tmp_PFM_path , const char * havana_executable) {
  char * target_file = util_alloc_filename( run_path , group->group_name , NULL);
  subst_list_filter_file( subst_list , group->modify_template , target_file);
  fault_group_link_faults(group , PFM_path , tmp_PFM_path);
  fault_group_fprintf_faultlist(group , tmp_PFM_path);

  {
    char * stdout_file = util_alloc_filename(run_path , "havana_stdout" , NULL);
    char * stderr_file = util_alloc_filename(run_path , "havana_stderr" , NULL);
    util_vfork_exec ( havana_executable , 1 , (const char *[1]) {target_file} , true , NULL , run_path , NULL , stdout_file , stderr_file);
    free(stderr_file);
    free(stdout_file);
  }

  free( target_file );
}



/*****************************************************************/




/* Transform */
void havana_fault_config_transform(const havana_fault_config_type * config , const double * input_data , double * output_data) 
{
    gen_kw_config_transform(config->gen_kw_config ,  input_data ,  output_data);
}



/**
   This function allocates a havana_fault_config instance. The primary
   role of this function is to allocate the actual object, *AND* to
   ensure that all the internal pointers are correctly initialized. 

   The state of havana_fault_config instance can then later be checked
   in havana_fault_config_assert().
*/



  
static havana_fault_config_type * __havana_fault_alloc( ) {
  havana_fault_config_type * config = util_malloc(sizeof * config , __func__);

  config->num_fault_groups  	= 0;
  config->fault_groups      	= NULL;
  config->havana_executable 	= NULL;
  config->input_fault_path      = NULL;
  config->update_template       = NULL;
  config->unfaulted_GRDECL_file = NULL;
  config->gen_kw_config         = NULL;
  config->__type_id             = HAVANA_FAULT_CONFIG_TYPE_ID;
  return config;
}


void havana_fault_config_set_executable(havana_fault_config_type * config , const char* executable) {
  config->havana_executable = util_realloc_string_copy(config->havana_executable , executable); 
}

void havana_fault_config_set_input_fault_path(havana_fault_config_type * config , const char* input_fault_path) {
  if (! util_is_directory(input_fault_path)) 
    util_abort("%s: INPUT_FAULTS must point to an existing directory.\n",__func__);

  config->input_fault_path = util_realloc_string_copy(config->input_fault_path , input_fault_path); 
}

void havana_fault_config_set_update_template(havana_fault_config_type * config , const char* update_template) {
  if (! util_file_exists(update_template))
    util_abort("%s: UPDATE must point to an existing file.\n",__func__);
  config->update_template = util_realloc_string_copy(config->update_template , update_template); 
}

void havana_fault_config_set_unfaulted_GRDECL_file(havana_fault_config_type * config , const char* GRDECL_file) {
  if (! util_file_exists(GRDECL_file))
      util_abort("%s: INPUT_ECLIPSE must point to an existing file.\n",__func__);
  config->unfaulted_GRDECL_file = util_realloc_string_copy(config->unfaulted_GRDECL_file , GRDECL_file); 
}


void havana_fault_config_add_fault_group(havana_fault_config_type * config , const char * group_name , const char * modify_template , int group_size , const char ** fault_names) {
  if (! util_file_exists(modify_template)) 
    util_abort("%s: template:%s does not exist. \n",__func__ , modify_template);

  if (config->input_fault_path == NULL) 
    util_abort("%s: must set the input_fault_path with INPUT_FAULTS before you can define fault groups\n",__func__);

  config->num_fault_groups++;
  config->fault_groups = util_realloc(config->fault_groups , config->num_fault_groups * sizeof * config->fault_groups , __func__);
  config->fault_groups[config->num_fault_groups - 1] = fault_group_alloc(group_name , group_size , fault_names , modify_template);
}


void havana_fault_config_set_gen_kw_config(havana_fault_config_type * config , const char * gen_kw_config_file) {
  if (config->gen_kw_config != NULL) {
    fprintf(stderr,"%s: ** Warning: freeing existsing gen_kw_config object from second DATA statement.\n",__func__);
    gen_kw_config_free(config->gen_kw_config);
  }
  config->gen_kw_config = gen_kw_config_fscanf_alloc( NULL , gen_kw_config_file , NULL , NULL );
}



/*
  Observe that the running of havana is a two-step process:

    1. Run havana for all the faultgroups (one at a time) with the
       havana action ModifyPFM.

    2. Run havana for all the faults with the intoeclipse action.
    
*/
void havana_fault_config_run_havana(const havana_fault_config_type * config , scalar_type * scalar_data , const char * run_path) {
  char * tmp_fault_input_path  = util_alloc_filename(run_path  , "tmp_havana_input_faults" , NULL);
  char * tmp_fault_output_path = util_alloc_filename(run_path  , "tmp_havana_output_faults" , NULL);
  const int size = havana_fault_config_get_data_size(config);
  int igroup;
  subst_list_type * subst_list = subst_list_alloc();

  scalar_transform( scalar_data );
  {
    const double * data = scalar_get_output_ref( scalar_data );
    int ikw;
    for (ikw = 0; ikw < size; ikw++) {
      char * tagged_fault = enkf_util_alloc_tagged_string(havana_fault_config_get_name(config , ikw));
      subst_list_insert_owned_ref(subst_list , tagged_fault , util_alloc_sprintf("%g" , data[ikw]));
      free(tagged_fault);
    }
  }
  {
    char * input_faults  = enkf_util_alloc_tagged_string("INPUT_FAULTS");
    char * output_faults = enkf_util_alloc_tagged_string("OUTPUT_FAULTS");
    char * input_eclipse = enkf_util_alloc_tagged_string("INPUT_ECLIPSE");
    
    subst_list_insert_owned_ref( subst_list , input_faults   , tmp_fault_input_path);
    subst_list_insert_owned_ref( subst_list , output_faults  , tmp_fault_output_path);
    subst_list_insert_ref( subst_list       , input_eclipse  , config->unfaulted_GRDECL_file);
    free(input_faults);
    free(output_faults);
    free(input_eclipse);
  }
  util_make_path(tmp_fault_input_path);
  util_make_path(tmp_fault_output_path);
  
  /* Run havana for each fault group to update displacement */
  /* Following is requirement to the template files: 
     1. One template file per fault group containing the havana ACTION ModifyPFM
     2. One template file for the ACTION IntoEclipse
     3. INPUT_FAULTS  must be specified by tmp_havana_input_faults  in template file for ACTION ModifyPFM
     4. OUTPUT_FAULTS must be specified by tmp_havana_output_faults in template file for ACTION ModifyPFM
     5. INPUT_FAULTS  must be specified by tmp_havana_output_faults in template file for ACTION IntoEclipse
     6. INPUT_ECLIPSE must be specified by <INPUT_ECLIPSE> in template file for ACTION IntoEclipse
  */
  /* In this program input faults will be read from temporary directory:   tmp_havana_input_faults */
  /* and output faults should be written to temporary directory: tmp_havana_output_faults */
  

  for (igroup = 0; igroup < config->num_fault_groups; igroup++) 
    fault_group_run_havana( config->fault_groups[igroup] , subst_list , run_path , config->input_fault_path , tmp_fault_input_path , config->havana_executable);


  /* Before running havana action IntoEclipse, we must set .faultlist to all faults in tmp_havana_output_faults */
  fault_group_fprintf_ALL_faultlist( (const fault_group_type **) config->fault_groups , config->num_fault_groups , tmp_fault_output_path);
  {
    char * target_file = util_alloc_filename( run_path , "update"  , NULL);
    char * stdout_file = util_alloc_filename( run_path , "havana_stdout"  , NULL);
    char * stderr_file = util_alloc_filename( run_path , "havana_stderr"  , NULL);
    
    subst_list_filter_file( subst_list , config->update_template , target_file);
    util_vfork_exec ( config->havana_executable , 1 , (const char *[1]) {target_file} , true , NULL , run_path , NULL , stdout_file , stderr_file);
    
    free(target_file);
    free(stdout_file);
    free(stderr_file);
  }
  subst_list_free(subst_list);
}


void havana_fault_config_printf(const havana_fault_config_type * config) {
  printf("Havana executable.........: %s \n",config->havana_executable);
  printf("Unfaulted grid............: %s \n",config->unfaulted_GRDECL_file);
  printf("Input faults..............: %s \n",config->input_fault_path);
  printf("Template for IntoEclipse..: %s \n\n",config->update_template);
}



/**
   Should check that the object is in a consistent state; essentially
   checking that all necessary variables have been set to a value !=
   NULL.
*/
static void havana_fault_config_assert(const havana_fault_config_type * config) {
  return;
  /*
    havana_fault_config_printf(config);
  */
}






/**
The format of the file should be as follows:

DATA             /tmp/Synthetic/sharedfiles/configfiles/havana_config.dat
INPUT_ECLIPSE    /tmp/Synthetic/sharedfiles/structureinput/Seismic_B_nothrow.GRDECL
INPUT_FAULTS     /tmp/Synthetic/sharedfiles/structureinput/PFM
UPDATE           /tmp/Synthetic/sharedfiles/templates/updateGrid.template
HAVANA           /d/proj/bg/enkf/EnKF_Structural/Havana/havana
GROUP            G01   /tmp/Synthetic/sharedfiles/templates/modifyPFM_G01.template Seismic_B_BaseThrow 
GROUP            G02   /tmp/Synthetic/sharedfiles/templates/modifyPFM_G02.template FaultX FaultY FaultZ
GROUP            G03   /tmp/Synthetic/sharedfiles/templates/modifyPFM_G03.template FaultA FaultB FaultC FaultD

Each group can contain many faults; but each fault must be in only one
group. This is not checked for. Note that the last group must
correspond to all faults that have no change in displacement

Observe that the OUTPUT_ECLIPSE variable should be relative to the
ECLIPSE run_path. The remaining variables can be both relative to the
CWD and absolute.
*/


#define ASSERT_TOKENS(kw,t,n) if ((t - 1) < (n)) util_abort("%s: when parsing %s must have at least %d arguments - aborting \n",__func__ , kw , (n)); 
havana_fault_config_type * havana_fault_config_fscanf_alloc(const char * filename) {
  havana_fault_config_type * config = __havana_fault_alloc();
  FILE  * stream = util_fopen(filename , "r");
  bool at_eof = false;


  while ( !at_eof ) {
    char  * line;
    int     tokens;
    char ** token_list;
    line  = util_fscanf_alloc_line(stream , &at_eof);
    if (line != NULL) {
      char * kw;
      util_split_string(line , " " , &tokens , &token_list);
      if (tokens > 0) {
	kw = token_list[0];
	if (strcmp(kw , "INPUT_ECLIPSE") == 0) {
	  ASSERT_TOKENS(kw , tokens , 1);
	  havana_fault_config_set_unfaulted_GRDECL_file(config , token_list[1]);
	} else if (strcmp(kw , "OUTPUT_ECLIPSE") == 0) {
	  ASSERT_TOKENS(kw , tokens , 1);
	  /*havana_fault_config_set_faulted_GRDECL_file(config , token_list[1]);*/
	} else if (strcmp(kw , "INPUT_FAULTS") == 0) {
	  ASSERT_TOKENS(kw , tokens , 1);
	  havana_fault_config_set_input_fault_path(config , token_list[1]);
	} else if (strcmp(kw , "UPDATE") == 0) {
	  ASSERT_TOKENS(kw , tokens , 1);
	  havana_fault_config_set_update_template(config , token_list[1]);
	} else if (strcmp(kw , "HAVANA") == 0) {
	  ASSERT_TOKENS(kw , tokens , 1);
	  havana_fault_config_set_executable(config , token_list[1]);
	} else if (strcmp(kw , "DATA") == 0) {
	  ASSERT_TOKENS(kw , tokens , 1);
	  havana_fault_config_set_gen_kw_config(config , token_list[1]);
	} else if (strcmp(kw , "GROUP") == 0) {
	  ASSERT_TOKENS(kw , tokens , 3);
	  {
	    const char  * group_name      = token_list[1];
	    const char  * modify_template = token_list[2];
	    const char ** fault_names     = (const char **) &token_list[3];
	    int group_size                = tokens - 3;        
	    havana_fault_config_add_fault_group(config , group_name , modify_template , group_size , fault_names);
	  }
	} else 
	  fprintf(stderr,"** Warning: keyword:%s in file:%s not recognized - ignored \n",kw , filename);
	util_free_stringlist(token_list , tokens);
      }
      free(line);
    }
  }
  fclose(stream);
  havana_fault_config_assert(config);
  return config;
}
#undef ASSERT_TOKENS




void havana_fault_config_free(havana_fault_config_type * havana_fault_config) 
{
    assert(havana_fault_config);
    assert(havana_fault_config->gen_kw_config);
    gen_kw_config_type *tmp = havana_fault_config->gen_kw_config;
    gen_kw_config_free(tmp);
    havana_fault_config->gen_kw_config = NULL;
    
    free(havana_fault_config->havana_executable);
    free(havana_fault_config->input_fault_path);
    free(havana_fault_config->unfaulted_GRDECL_file);
    free(havana_fault_config->update_template);
    {
      int igroup;
      for (igroup = 0; igroup < havana_fault_config->num_fault_groups; igroup++)
	fault_group_free(havana_fault_config->fault_groups[igroup]);
      free(havana_fault_config->fault_groups);
    }
    free(havana_fault_config);
}



int havana_fault_config_get_data_size(const havana_fault_config_type * havana_fault_config) 
{
    return (gen_kw_config_get_data_size(havana_fault_config->gen_kw_config));
}


const char * havana_fault_config_get_name(const havana_fault_config_type * config, int kw_nr) 
{
    return (gen_kw_config_iget_name(config->gen_kw_config,  kw_nr));
}


char ** havana_fault_config_get_name_list(const havana_fault_config_type * config) 
{
  return ( gen_kw_config_get_name_list(config->gen_kw_config));
}

const char * havana_fault_config_get_template_ref(const havana_fault_config_type * config) 
{
    return ( gen_kw_config_get_template_ref(config->gen_kw_config));
}

void  havana_fault_config_truncate(const havana_fault_config_type * config, scalar_type * scalar)
{
    util_abort("%s: the function is not implemented\n",__func__);
    /*    gen_kw_config_truncate(config->gen_kw_config, scalar); */
}


gen_kw_config_type * havana_fault_config_get_gen_kw_config(const havana_fault_config_type * config) {
  return config->gen_kw_config;
}



/*****************************************************************/

SAFE_CAST(havana_fault_config , HAVANA_FAULT_CONFIG_TYPE_ID)
VOID_FREE(havana_fault_config)
VOID_GET_DATA_SIZE(havana_fault)





