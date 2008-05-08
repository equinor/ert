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

  if (argc < 7) {
    printf("Usage: test_havana   <directoryprefix> <executable>  <template_model_file>  <configfile> <target model file> <nreal>\n");
    exit(1);
  }

  /*

  o Ikke bland variabler og kode - det er ryddig og bra med variabler
    som har begrenset leksikalt skope, men bruk i det tilfellet { }
    for å tydeliggjøre skopet.

  o I slike opplistinger som er gitt nedenfor så bør vi ta oss tid til
    å opplinjere på "=" - det øker lesbarheten.

  o Jeg har startet med 'directory_prefix' konvensjonen; jeg skal ikke
    argumentere for at den ene konvensjonen er bedre enn den
    andre. Men, jeg vil argumentere TUNGT for at en konvensjon er
    bedre enn flere. Jeg ber derfor om at vi ikke bruker computerCaps.

  o atoi() er selvfølgelig OK, men funksjonen util_sscanf_int() er
    basert på funksjonen strtol() som er en forbedring av atoi(), den
    gir også (mulighet for) feilhåndtering.

  o Noen ganger er det selvfølgelig svært fristende å allokere en
    streng som er "lang nok"; i dette tilfellet bør det ikke være
    nødvendig: 
     * Bruk strlen() til å allokere en streng av riktig lengde.
     * Bruk util_malloc() til å allokere - da får du automatisk en
       sjekk på at allokeringen gikk bra.
     * Ikke cast retur-verdien fra malloc() - det gir ingenting.
     * Når det gjelder filnavn / kataloger så forsøk å bruke
       util_path_xxx() relaterte funksjoner, da blir det (eventuelt)
       mye lettere å porte til windows på et senere tidspunkt.
     * Variabelen format skal kun leve som input til en path_fmt
       instans - den bør defineres som en variabel med begrenset
       skope. 
  */
  
  {
    char * directory_prefix    = argv[1];
    char * executable_file     = argv[2];
    char * template_model_file = argv[3];
    char * config_file         = argv[4];
    char * target_model_file   = argv[5];
    int    ens_size;

    path_fmt_type             * run_path_fmt;
    havana_fault_type        ** ensemble; 
    havana_fault_config_type  * havana_config;

    if (!util_sscanf_int(argv[6] , &ens_size)) {
      fprintf(stderr,"Failed to interpret:%s as an integer - exiting.\n",argv[6]);
      exit(1);
    }
    {
      const char * file_fmt = "ens%d";
      char       * format   = util_alloc_full_path(directory_prefix , file_fmt);
      run_path_fmt = path_fmt_alloc_directory_fmt(format,true);
      free(format);
    }
    havana_config =  havana_fault_config_fscanf_alloc(config_file, template_model_file , executable_file);
    ensemble      =  util_malloc(ens_size * sizeof *ensemble , __func__);
    
    {
      /* 
        Bruk leksikalt begrensede telle-variabler, forsøk å gi
        tellevariblene et navn som indikerer hva de teller over. Det
        er for eksempel åpenbart at 'iens' teller over ensemble
        medlemmer.
      */
      int iens;

      for (iens=0; iens < ens_size; iens++) {
	  
	  char * target_file = path_fmt_alloc_file(run_path_fmt , iens+1 ,target_model_file);
	  
	  ensemble[iens] = havana_fault_alloc(havana_config);
	  
	  havana_fault_initialize(ensemble[iens],iens);
	  
	  havana_fault_ecl_write(ensemble[iens] , target_file);
	  
	  free(target_file);
	  
	  havana_fault_free(ensemble[iens]); 
      }
    }
    free(ensemble);
    havana_config_free(havana_config);  
    path_fmt_free(run_path_fmt);
  }
}   




/* int main(int argc, char ** argv)  */
/* { */

/*     if(argc < 7) */
/*       { */
/* 	printf("Usage: test_havana   <directoryprefix> <executable>  <template_model_file>  <configfile> <target model file> <nreal>\n"); */
/* 	exit(1); */
/*       } */
/*     char * directoryPrefix = argv[1]; */
/*     char * executableFile = argv[2]; */
/*     char * templateModelFile = argv[3]; */
/*     char * configFile = argv[4]; */
/*     char * targetModelFile = argv[5]; */
/*     int    ens_size = atoi(argv[6]); */

/*     char * format = (char *) malloc(200); */

/*     havana_fault_type ** ensemble;  */

/*     sprintf(format,"%s/ens%s",directoryPrefix,"%d"); */
/*     printf("%s\n",format); */

/*     path_fmt_type * run_path_fmt = path_fmt_alloc_directory_fmt(format,true); */





/*    havana_fault_config_type * havana_config =  havana_fault_config_fscanf_alloc(configFile, templateModelFile,executableFile); */

/*    int i; */

   

/*    ensemble = (havana_fault_type **) util_malloc(ens_size * sizeof *ensemble , __func__); */

/*    for (i=0; i < ens_size; i++)  */
/*    {  */

/*       char * target_file = path_fmt_alloc_file(run_path_fmt , i+1 ,targetModelFile); */

/*       ensemble[i] = havana_fault_alloc(havana_config); */

/*       havana_fault_initialize(ensemble[i],i); */

/*       havana_fault_ecl_write(ensemble[i] , target_file); */

/*       free(target_file); */

/*       havana_fault_free(ensemble[i]);  */

/*    } */

/*    free(ensemble); */

/*    path_fmt_free(run_path_fmt); */

/* }    */

