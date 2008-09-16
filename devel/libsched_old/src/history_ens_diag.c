#include <string.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <dirent.h>

#include <set.h>
#include <hash.h>
#include <ecl_fstate.h>
#include <util.h>
#include <ecl_sum.h>
#include <history_ens_diag.h>
#include <history.h> /* Ugly - circumventing dependencies for a quick hack. */


static const char * alloc_wellvar_name(const char *path , const char *well , const char *var) {
  char *out_file = malloc(strlen(path) + 1 + strlen(well) + 1 + strlen(var) + 1);
  sprintf(out_file , "%s/%s.%s" , path , well , var);
  return out_file;
}


static void set_well_var(const char *file , char **_well , char **_var) {
  char *var;
  var = strrchr(file , '.') + 1;
  *_var = malloc(strlen(var) + 1);
  strcpy(*_var , var);


  {
    int n = 0;
    while (file[n] != '.')
      n++;
    
    *_well = malloc(n+1);
    strncpy(*_well , file , n);
    (*_well)[n] = '\0';
  }
}


void history_ens_diag_imake_plotfile(int iens1 , int iens2 , int min_size , const ecl_sum_type **ecl_sum_list , const char *out_file , const char * title , const char * well , const char * var, int index , history_type * history) {
  FILE *stream;
  int iens,istep,first_report;

  
  stream = util_fopen(out_file , "w");
  util_fwrite_string(title , stream); 					       /* Plot title */
  util_fwrite_string(var   , stream); 					       /* Variable name */                    
  util_fwrite_string(ecl_sum_get_unit_ref(ecl_sum_list[0] , var) , stream);    /* Unit */
  util_fwrite_int(iens2 - iens1 + 1 , stream);  			       /* Ensemble size */
  util_fwrite_int(min_size , stream);           			       /* Member length */         

  {
    int tmp;
    ecl_sum_get_report_size( ecl_sum_list[0] , &first_report , &tmp);
  }
  for (istep=first_report; istep < ( first_report + min_size); istep++) {
    int date[3];
    time_t sim_time = ecl_sum_get_sim_time(ecl_sum_list[0] , istep );
    if (sim_time == -1) {
      date[0] = 0; date[1] = 0; date[2] = 0;
      fprintf(stderr,"** Warning: day/month/year not defined in summary_files. \n");
    } else 
      util_set_date_values(ecl_sum_get_sim_time(ecl_sum_list[0] , istep) , &date[0] , &date[1] , &date[2]);
    
    util_fwrite_int_vector(date , 3 , stream , __func__);  /* True time  day/month/year */
  }

  for (istep=0; istep < min_size; istep++) {
    double history_value = 0;
    if (istep > 0)
      if (history != NULL)
	history_value = history_get(history , istep, well , var);
    
    util_fwrite_double(history_value , stream);                              /* History value */ 
  }
  
  for (istep = 0; istep < min_size;  istep++) {                              /* Ensemble values: ensemble direction is the fast index */
    for (iens = iens1; iens <= iens2; iens++) 
      util_fwrite_double(ecl_sum_iget_with_index(ecl_sum_list[iens - iens1]  , istep , index) , stream);
    
  }
  fclose(stream);
}



static void history_ens_diag_make_plotfile(int iens1 , int iens2 , int min_size , const ecl_sum_type **ecl_sum_list , const char *out_path , const char *well , const char *var, history_type * history) {
  char *hvar           = util_malloc(strlen(var) + 2 , __func__ );
  const char *out_file = alloc_wellvar_name(out_path , well , var);
  char * title         = util_malloc(strlen(well) + 3 + strlen(var) + 1 , __func__); 
  int index;
  
  if (!ecl_sum_has_well_var(ecl_sum_list[0] , well , var)) {
    fprintf(stderr,"No data for %s/%s \n",well , var);
    return;
  }
  sprintf(hvar     , "%sH" , var);
  sprintf(title , "%s : %s" , well , var);
  index = ecl_sum_get_well_var_index(ecl_sum_list[0] , well , var);
  history_ens_diag_imake_plotfile(iens1 , iens2 , min_size , ecl_sum_list , out_file , title , well , var , index , history);
  
  free((char *) out_file);
  free(hvar);
}


ecl_sum_type ** history_ens_diag_load_ensemble2(int iens1, int iens2 , int min_size , int *_ens_size , const char *ens_path , const char *eclbase_dir , const char *eclbase , bool report_mode ,  bool fmt_file, bool unified , bool endian_convert , history_type * history) {
  int ens_size = iens2 - iens1 + 1;
  int _min_size;
  ecl_sum_type ** ecl_sum_list2;
  ecl_sum_type ** ecl_sum_list1 = history_ens_diag_load_ensemble(iens1 , iens2 , &_min_size , ens_path , eclbase_dir , eclbase , report_mode , fmt_file , unified , endian_convert , history);
  if (_min_size < min_size) {
    int iens;
    ens_size = 0;
    for (iens = iens1; iens < iens2; iens++)
      if (ecl_sum_get_size(ecl_sum_list1[iens]) >= min_size)
	ens_size++;

    ecl_sum_list2 = util_malloc(ens_size * sizeof * ecl_sum_list2 , __func__);
    {
      int local_iens = 0;
      for (iens = iens1; iens < iens2; iens++)
	if (ecl_sum_get_size(ecl_sum_list1[iens]) >= min_size) {
	  ecl_sum_list2[local_iens] = ecl_sum_list1[iens];
	  local_iens++;
	} else ecl_sum_free(ecl_sum_list1[iens]);
    }
    *_ens_size= ens_size;
    free(ecl_sum_list1);
    printf("Reduced size: %d \n",ens_size);
    return ecl_sum_list2;
  } else abort();
}


ecl_sum_type ** history_ens_diag_load_ensemble(int iens1, int iens2 , int * _min_size , const char *ens_path , const char *eclbase_dir , const char *eclbase , bool report_mode ,  bool fmt_file, bool unified , bool endian_convert , history_type * history) {
  ecl_sum_type **ecl_sum_list = calloc(iens2 - iens1 + 1 , sizeof(ecl_sum_type *));
  int iens;
  int fmt_mode , min_size;
  if (fmt_file)
    fmt_mode = ECL_FORMATTED;
  else
    fmt_mode = ECL_BINARY;

  for (iens = iens1; iens <= iens2; iens++) {
    char * spec_file;
    char * path = malloc(strlen(ens_path) + 1 + strlen(eclbase_dir) + 4 + 1);
    char base[256];
    sprintf(base , "%s-%04d" , eclbase , iens);
    /*char * base = malloc(strlen(eclbase) + 1 + 4 + 1 );*/
    /*sprintf(base , "%s-%04d"  , eclbase , iens);*/
    sprintf(path , "%s/%s%04d" , ens_path, eclbase_dir , iens);
    /*
      base = ecl_util_alloc_base_guess(path);
      if (base == NULL) {
      abort();
      }
    */
    spec_file = ecl_util_alloc_exfilename(path , base , ecl_summary_header_file , fmt_file , -1);
    
    if (unified) {
      char *data_file = ecl_util_alloc_exfilename(path , base , ecl_unified_summary_file , fmt_file , -1);
      printf("Loading file: %s ... ",data_file); fflush(stdout);
      ecl_sum_list[iens - iens1] = ecl_sum_fread_alloc(spec_file , 1 , (const char **) &data_file , report_mode , endian_convert);
      printf("%d timestep \n",ecl_sum_get_size(ecl_sum_list[iens - iens1]));
      free(data_file);
    } else {
      int files;
      char **fileList;
      fileList = ecl_util_alloc_scandir_filelist(path , base , ecl_summary_file , fmt_file , &files); 
      printf("Loading from directory: %s ... ",path); fflush(stdout);
      ecl_sum_list[iens - iens1] = ecl_sum_fread_alloc(spec_file , files , (const char **) fileList , report_mode , endian_convert);
      printf("%d timestep \n",ecl_sum_get_size(ecl_sum_list[iens - iens1]));
      util_free_stringlist(fileList , files);
    }

    free(spec_file);
    free(path);
    /*free(base);*/
  }
  {
    bool size_eq = true;
    int size0    = ecl_sum_get_size(ecl_sum_list[0]);
    int iens;
    min_size = size0;
    for (iens = 1; iens <= (iens2 - iens1); iens++) {
      int size = ecl_sum_get_size(ecl_sum_list[iens]);
      size_eq = size_eq && (size == size0);
      if (size < min_size)
	min_size = size;
    }
    if (! size_eq) 
      printf("Data cut at timestep:%d \n",min_size);
  }
  *_min_size = min_size;
  return ecl_sum_list;
}




static void history_ens_diag_ens(int iens1 , int iens2 , const char *out_path , int nwell , const char **well_list , int nvar , const char **var_list , const char *ens_path , const char *eclbase_dir , const char *eclbase, bool report_mode , bool fmt_file, bool unified, bool endian_convert , history_type * history) {
  int iwell,ivar,min_size;
  
  ecl_sum_type **ecl_sum_list = history_ens_diag_load_ensemble(iens1 , iens2 , &min_size , ens_path , eclbase_dir , eclbase , report_mode , fmt_file , unified , endian_convert , history);
  util_make_path(out_path);
  for (iwell = 0; iwell < nwell; iwell++) {
    for (ivar = 0; ivar < nvar; ivar++) {
      history_ens_diag_make_plotfile(iens1 , iens2 , min_size , (const ecl_sum_type **) ecl_sum_list , out_path , well_list[iwell] , var_list[ivar] , history);
    }
  }

  return ; /* Ignore the misfit calculations */
  {
    char **hvar_list;
    double *min , *max , *inv_covar;
    double *std , *well_misfit;
    int ivar , iens;
    
    hvar_list = malloc(nvar * sizeof * hvar_list);
    for (ivar = 0; ivar < nvar; ivar++) {
      hvar_list[ivar] = malloc(strlen(var_list[ivar]) + 2);
      sprintf(hvar_list[ivar] , "%sH" , var_list[ivar]);
    }
    
    min       	= malloc(nvar * sizeof  * min);
    max       	= malloc(nvar * sizeof  * max);
    std       	= malloc(nvar * sizeof  * std);
    inv_covar 	= malloc(nvar * nvar * sizeof * inv_covar);
    well_misfit = malloc(nwell * sizeof * well_misfit);
    
    for (iens = iens1; iens <= iens2; iens++)
      ecl_sum_max_min(ecl_sum_list[iens - iens1] , nwell , (const char **) well_list , nvar , (const char **) hvar_list , max , min , (iens == iens1));
    
    for (ivar = 0; ivar < nvar*nvar; ivar++) 
      inv_covar[ivar] = 0;

    for (ivar = 0; ivar < nvar; ivar++) {
      std[ivar] = 0.10 * max[ivar];
      inv_covar[ivar*(nvar + 1)] = 1.0 / (std[ivar] * std[ivar]);
    }
    
    {
      char * out_file = util_alloc_full_path(out_path , "misfit.txt");
      FILE * stream   = util_fopen(out_file , "w");

      fprintf(stream,"Model number  ");
      for (iwell = 0; iwell < nwell; iwell++)
	fprintf(stream,"  %10s    |" , well_list[iwell]);
      fprintf(stream,"  %10s\n" , "Total");
      fprintf(stream,"--------------");
      for (iwell = 0; iwell < nwell; iwell++)
	fprintf(stream,"----------------|");
      fprintf(stream,"---------------\n");;
      
      for (iens = iens1; iens <= iens2; iens++) {
	double total_misfit = ecl_sum_eval_misfit(ecl_sum_list[iens - iens1] , nwell , (const char **) well_list , nvar , (const char **) var_list , inv_covar , well_misfit);
	fprintf(stream,"%3d             ",iens);
	for (iwell=0; iwell < nwell; iwell++) 
	  fprintf(stream,"  %10.3f  |  ",well_misfit[iwell]);
	fprintf(stream,"%12.3f \n",total_misfit);
      }
      
      fprintf(stream,"--------------");
      for (iwell = 0; iwell < nwell; iwell++)
	fprintf(stream,"----------------|");
      fprintf(stream,"---------------\n");;
      
      fclose(stream);
      printf("Misfit information written to: %s \n",out_file);
      free(out_file);
    }
    free(well_misfit);
    free(min);
    free(max);
    free(inv_covar);
    free(std);
    util_free_stringlist(hvar_list , nvar);
  }
  
  {
    int i;
    for (i=0; i <= (iens2 - iens1); i++)
      ecl_sum_free(ecl_sum_list[i]);
  }
  free(ecl_sum_list);
}


static void print_prompt(const char *_prompt , int len) {
  char *prompt;
  if (len < strlen(_prompt))
    prompt = malloc(strlen(_prompt) + 5);
  else
    prompt = malloc(len + 5);
  strcpy(prompt , _prompt);
  {
    int N = len - strlen(prompt);
    int i;
    for (i=0; i < N; i++)
      strcat(prompt , ".");
  }
  strcat(prompt , ": ");
  printf(prompt);
  free(prompt);
}



static void read_bool(const char *prompt , int len , bool *yn) {
  char c;
  do {
    print_prompt(prompt , len);
    fscanf(stdin , "%c" , &c);
  } while (c != 'y' && c != 'Y' && c != 'n' && c != 'N');
  if (c == 'n' || c == 'N')
    *yn = false;
  else
    *yn = true;
}


static void read_string(const char *prompt , int len , char *s) {
  print_prompt(prompt , len);
  fscanf(stdin , "%s" , s);
}

static void read_int(const char *prompt , int len , int *i) {
  print_prompt(prompt, len); 
  fscanf(stdin , "%d" , i);
}



static char ** fread_alloc_wells(const char *well_file , int *_nwell) {
  FILE *fileH;
  char **well_list = NULL;
  char well[32];
  set_type * well_set = set_alloc_empty();

  if (util_file_exists(well_file)) {
    fileH = fopen(well_file , "r");
    util_fskip_lines(fileH , 3);
    while ( fscanf(fileH , "%s" , well ) == 1) {
      set_add_key(well_set , well);
      util_fskip_lines(fileH , 1);
    }
    fclose(fileH);
  }
  *_nwell = set_get_size(well_set);
  well_list = set_alloc_keylist(well_set);
  
  set_free(well_set);
  return well_list;
}






void history_ens_diag_ens_interactive(const char *eclbase_dir , const char *eclbase_name , bool fmt_file , bool unified , bool endian_convert, history_type * history) {
#define defvar_N 3
  const bool report_mode = true;
  const int prompt_len   = 68;
  char out_path[128];
  char ens_path[128];
  int  i , iens1,iens2,nwell,nvar;
  char prompt[128];
  const char *defvar_list[defvar_N] = {"WOPR" , "WGOR" , "WWCT"};
  char **well_list;
  char **var_list;

  printf("eclbase_name:%s \n",eclbase_name);
  printf("-----------------------------------------------------------------\n");
  read_string("Path to ensemble" , prompt_len , ens_path);
  if (!util_file_exists(ens_path)) {
    fprintf(stderr,"Warning path: %s does not exist - returning. \n",ens_path);
    return;
  }
  
  read_int   ("First ensemble member " , prompt_len , &iens1);
  read_int   ("Last ensemble member " , prompt_len , &iens2);

  read_int   ("Number of wells ...[0 to use all wells in prepobs.def]" , prompt_len , &nwell);
  if (nwell == 0) {
    well_list = fread_alloc_wells("prepobs.def" , &nwell);
    if (nwell == 0) {
      fprintf(stderr,"Could not find well file: prepobs.def - returning \n");
      return;
    }
  } else {
    well_list = util_alloc_stringlist(nwell , 64);
    for (i=0; i < nwell; i++) {
      sprintf(prompt , "   Well %2d" , i+1);
      read_string(prompt , prompt_len , well_list[i]);
    }
  }
  
  read_int ("Number of variables  ...[0 to use default set: WOPR WGOR WWCT]" , prompt_len , &nvar);
  if (nvar == 0) {
    nvar = defvar_N;
    var_list = util_alloc_stringlist(nvar , 64);
    for (i=0; i < nvar; i++) 
      strcpy(var_list[i] , defvar_list[i]);
  } else {
    var_list = util_alloc_stringlist(nvar , 64);
    for (i=0; i < nvar; i++) {
      sprintf(prompt , "   Var %2d                " , i+1);
      read_string(prompt , prompt_len , var_list[i]);
    }
  }

  read_string("Path to store the results" , prompt_len , out_path);
  /*
    read_bool("Add tecplot header" , prompt_len , &tecplot);
  */
  history_ens_diag_ens(iens1 , iens2 , out_path , nwell , (const char **) well_list , nvar , (const char **) var_list , ens_path , eclbase_dir , eclbase_name , report_mode , fmt_file , unified , endian_convert , history);



  util_free_stringlist(well_list , nwell);
  util_free_stringlist(var_list  , nvar);
}


