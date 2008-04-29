#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <havana_fault_config.h>
#include <havana_fault.h>
#include <path_fmt.h> 
#include <util.h>

int main(int argc, char ** argv) 
{

    if(argc < 7)
        {
            printf("Usage: test_havana   <directoryprefix> <executable>  <template_model_file>  <configfile> <target model file> <nreal>\n");
            exit(1);
        }
    char * directoryPrefix = argv[1];
    char * executableFile = argv[2];
    char * templateModelFile = argv[3];
    char * configFile = argv[4];
    char * targetModelFile = argv[5];
    int    ens_size = atoi(argv[6]);

    char * format = (char *) malloc(200);
    sprintf(format,"%s/ens%s",directoryPrefix,"%d");
    printf("%s\n",format);

    path_fmt_type * run_path_fmt = path_fmt_alloc_directory_fmt(format,true);



   havana_fault_type ** ensemble; 

   havana_fault_config_type * havana_config =  havana_fault_config_fscanf_alloc(configFile, templateModelFile,executableFile);

   int i;

   

   ensemble = (havana_fault_type **) util_malloc(ens_size * sizeof *ensemble , __func__);

   for (i=0; i < ens_size; i++) 
   { 

      char * target_file = path_fmt_alloc_file(run_path_fmt , i+1 ,targetModelFile);

      ensemble[i] = havana_fault_alloc(havana_config);

      havana_fault_initialize(ensemble[i],i);

      havana_fault_ecl_write(ensemble[i] , target_file);

      free(target_file);

      havana_fault_free(ensemble[i]); 

   }

   free(ensemble);

   path_fmt_free(run_path_fmt);

}   
