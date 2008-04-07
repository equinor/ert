#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <enkf_util.h>
#include <util.h>
#include <ens_config.h>
#include <enkf_macros.h>
#include <relperm_config.h>
#include <util.h>
#include <trans_func.h>
#include <hash.h>
#include <math.h>

static relperm_config_type * __relperm_config_alloc_empty( int size_conf, int size_tab ) {
  
  relperm_config_type *relperm_config = malloc(sizeof *relperm_config);

  relperm_config->scalar_config = scalar_config_alloc_empty(size_conf);
  relperm_config->var_type = parameter;

  relperm_config->nso           = 0;
  relperm_config->nsw           = 0;
  relperm_config->nsg           = 0;
  relperm_config->famnr         = 0;
  relperm_config->kw_list       = enkf_util_malloc(size_conf * sizeof  *relperm_config->kw_list, __func__);
  relperm_config->index_hash    = hash_alloc(10);
  relperm_config->ecl_file_hash = hash_alloc(10);
  
  relperm_config->table_config = enkf_util_malloc(size_tab * sizeof *relperm_config->table_config,__func__); 
  
  return relperm_config;
}

relperm_config_type * relperm_config_fscanf_alloc(const char * config_file, const char * table_file){
  relperm_config_type * config;
  int size_conf,size_tab, line_nr;

  FILE * stream_config = util_fopen(config_file, "r");
  FILE * stream_table = util_fopen(table_file,"r");
  bool at_eof = false;

  size_conf = util_count_file_lines(stream_config);
  fseek(stream_config, 0L, SEEK_SET);

  size_tab = util_count_file_lines(stream_table);  
  fseek(stream_table, 0L, SEEK_SET);

  config = __relperm_config_alloc_empty(size_conf,size_tab-1);
  config->num_tab = size_tab-1;
  line_nr = 0;
  do {
    char name[128];
    if (fscanf(stream_config,"%s" , name) != 1) {
      fprintf(stderr,"%s: Something wrong with the input in relperm_config \n",__func__);
      abort();
    }
    config->kw_list[line_nr] = util_alloc_string_copy(name);
    scalar_config_fscanf_line(config->scalar_config, line_nr , stream_config);
    printf("kw_list %s \n",config->kw_list[line_nr]);

    if(!hash_has_key(config->index_hash,config->kw_list[line_nr])){    
      hash_insert_int(config->index_hash,config->kw_list[line_nr],line_nr);
    }
    else{
      fprintf(stderr,"Problem in %s: The keyword %s must be unique in relperm_config.txt \n",__func__,config->kw_list[line_nr]);
      abort();
    }
    printf("hash_get_int %d \n",hash_get_int(config->index_hash,config->kw_list[line_nr]));
    line_nr++;
  }while(line_nr < size_conf);


  if(fscanf(stream_table,"%i %i %i %i",&config->nso,&config->nsw,&config->nsg,&config->famnr) != 4) {
    fprintf(stderr,"%s: Something wrong with the input in line one of relperm_table.txt \n",__func__);
    abort();
  }
  /* Sanity check on grids */
  if(config->nso <= 1 || config->nsw <= 1 || config->nsg <= 1){
    fprintf(stderr,"%s: Error: nso,nsw and nsg must be at least 2 \n",__func__);
  } 

  /* Allocation of table input in relperm_table.txt */
  util_forward_line(stream_table, &at_eof);    
  line_nr = 0;
  printf("size_tab and config-num_tab %d %d\n",size_tab,config->num_tab);
  do{
    config->table_config[line_nr]=relperm_config_fscanf_table_config_alloc(stream_table);
    relperm_config_check_tab_input(config->index_hash,config->table_config[line_nr],config->famnr);
    printf("config->table_config->eclipse_file %s \n",config->table_config[line_nr]->eclipse_file);
    line_nr++;
  }while(line_nr < size_tab-1);  

  fclose(stream_table);
  fclose(stream_config);

  printf("config->table_config->tab_corey->swco %s \n",config->table_config[0]->tab_corey->swco);

  return config;
}

table_config_type * relperm_config_fscanf_table_config_alloc(FILE * stream_tab ){
  char ** token_list;
  int tokens;
  char * line;
  bool at_eof = false;

  table_config_type * tab_config = util_malloc(sizeof *tab_config,__func__);      

  printf("Er i relperm_config_fscanf_table_alloc \n");
  printf("Har lest argumenter:%d \n",ftell(stream_tab));

  line = util_fscanf_alloc_line(stream_tab,&at_eof);
  printf("Skrive line: %s \n",line);

  if (line != NULL){
    util_split_string(line," ", &tokens, &token_list);
    /* tokens has to be equal to 16 or else the line in relperm_table.txt is not correctly specified */ 
    if(tokens == 16){
      tab_config->eclipse_file = util_alloc_string_copy(token_list[0]);
      tab_config->relptab_kw   = relperm_config_set_relptab_kw(token_list[1]);
    }
    else{
      printf("The line in relperm_table.txt is not correctly specified \n");
      fprintf(stderr,"%s: The number of arguments should be 16, and the line has %d arguments",__func__,tokens);
      abort();
    }
  }
  else{
    fprintf(stderr,"%s: Something wrong when reading relperm_tab.txt",__func__);
    abort();
  }

  if(strcmp(token_list[2], "COREY") == 0){
    tab_config->func = COREY;
  }
  else{
    fprintf(stderr,"Problem in %s. The relperm function %s is not valid.",__func__,token_list[2]);
    abort();
  }

  switch(tab_config->func){
  case(COREY):
    {
      tab_config->tab_corey = relperm_config_table_corey_alloc(token_list,tokens);
      break;
    }  
  default:
    fprintf(stderr,"Problem in %s. Wrong input in relperm_tab.txt: Options are: COREY \n",__func__);
    abort();
  }

  util_free_string_list(token_list,tokens);
  free(line);
  return tab_config;
}

table_corey_type * relperm_config_table_corey_alloc(char ** token_list, int tokens){
    table_corey_type * corey = util_malloc(sizeof *corey,__func__);      
    corey->swco = util_alloc_string_copy(token_list[3]);
    corey->soco = util_alloc_string_copy(token_list[4]);
    corey->sgco = util_alloc_string_copy(token_list[5]);
    corey->sorg = util_alloc_string_copy(token_list[6]);
    corey->ewat = util_alloc_string_copy(token_list[7]);
    corey->egas = util_alloc_string_copy(token_list[8]);
    corey->eowa = util_alloc_string_copy(token_list[9]);
    corey->eogw = util_alloc_string_copy(token_list[10]);
    corey->scwa = util_alloc_string_copy(token_list[11]);
    corey->scga = util_alloc_string_copy(token_list[12]);
    corey->scoi = util_alloc_string_copy(token_list[13]);
    corey->pewa = util_alloc_string_copy(token_list[14]);
    corey->pega = util_alloc_string_copy(token_list[15]);
    return corey;
}

void relperm_config_check_tab_input(const hash_type * index_hash,const table_config_type * tab_config, const int famnr){
  printf("Er i relperm_config_check_tab_input \n");    
  
  if(famnr == 1){
    if(tab_config->relptab_kw==SWOF || tab_config->relptab_kw==SGOF || tab_config->relptab_kw==SLGOF){
      printf("Relperm keyword is consistent with the family nr %d \n",famnr);
    }
    else{
      fprintf(stderr,"%s: Relperm keyword is not consistent with the family nr %d \n",__func__,famnr);
      abort();
    }
  }
  if(famnr == 2){
    if(tab_config->relptab_kw ==SWFN || tab_config->relptab_kw ==SGFN || tab_config->relptab_kw ==SOF2 || tab_config->relptab_kw==SOF3 || tab_config->relptab_kw==SOF32D){
      printf("Relperm keyword is consistent with the family nr %d \n",famnr);
    }
    else{
      fprintf(stderr,"%s: Relperm keyword is not consistent with the family nr %d \n",__func__,famnr);
      abort();
    }    
  }

  switch(tab_config->func){
  case(COREY):
    {
      table_corey_type * corey = tab_config->tab_corey;
      if(!hash_has_key(index_hash,corey->swco)){
        fprintf(stderr,"%s: The keyword %s for corey->swco is not defined in relperm_config.txt \n",__func__,corey->swco);
	abort();
      }
      else if(!hash_has_key(index_hash,corey->soco)){
	fprintf(stderr,"%s: The keyword %s for corey->soco is not defined in relperm_config.txt \n",__func__,corey->soco);
	abort();
      }
      else if(!hash_has_key(index_hash,corey->sgco)){
	fprintf(stderr,"%s: The keyword %s for corey->sgco is not defined in relperm_config.txt \n",__func__,corey->sgco);
	abort();
      }
      else if(!hash_has_key(index_hash,corey->sorg)){
	fprintf(stderr,"%s: The keyword %s for corey->sorg is not defined in relperm_config.txt \n",__func__,corey->sorg);
	abort();
      }
      else if(!hash_has_key(index_hash,corey->ewat)){
	fprintf(stderr,"%s: The keyword %s for corey->ewat is not defined in relperm_config.txt \n",__func__,corey->ewat);
	abort();
      }
      else if(!hash_has_key(index_hash,corey->egas)){
	fprintf(stderr,"%s: The keyword %s for corey->egas is not defined in relperm_config.txt \n",__func__,corey->egas);
	abort();
      }
      else if(!hash_has_key(index_hash,corey->eowa)){
	fprintf(stderr,"%s: The keyword %s for corey->eowa is not defined in relperm_config.txt \n",__func__,corey->eowa);
	abort();
      }
      else if(!hash_has_key(index_hash,corey->eogw)){
	fprintf(stderr,"%s: The keyword %s for corey->eogw is not defined in relperm_config.txt \n",__func__,corey->eogw);
	abort();
      }
      else if(!hash_has_key(index_hash,corey->scwa)){
	fprintf(stderr,"%s: The keyword %s for corey->scwa is not defined in relperm_config.txt \n",__func__,corey->scwa);
	abort();
      }
      else if(!hash_has_key(index_hash,corey->scga)){
	fprintf(stderr,"%s: The keyword %s for corey->scga is not defined in relperm_config.txt \n",__func__,corey->scga);
	abort();
      }
      else if(!hash_has_key(index_hash,corey->scoi)){
	fprintf(stderr,"%s: The keyword %s for corey->scoi is not defined in relperm_config.txt \n",__func__,corey->scoi);
	abort();
      }
      else if(!hash_has_key(index_hash,corey->pewa)){
	fprintf(stderr,"%s: The keyword %s for corey->pewa is not defined in relperm_config.txt \n",__func__,corey->pewa);
	abort();
      }
      else if(!hash_has_key(index_hash,corey->pega)){
	fprintf(stderr,"%s: The keyword %s for corey->pega is not defined in relperm_config.txt \n",__func__,corey->pega);
	abort();
      }
      else{
      printf("Input is ok for table %s \n",tab_config->eclipse_file);    
      }
      break;
    }
  default:
    fprintf(stderr,"%s: The relperm function is not implemented \n",__func__);
    abort();
  }
}

int relperm_config_get_data_size(const relperm_config_type * relperm_config) {
  return scalar_config_get_data_size(relperm_config->scalar_config);
}


void relperm_config_ecl_write(const relperm_config_type * config, const double * data, const char * path){
  int ik;
  table_config_type ** table_config;
  bool ecl_file_append=false;

  table_config = config->table_config;

  for(ik =0; ik < config->num_tab;ik++){
    FILE * relp_ecl_stream;
    char * relpfile;

    relpfile = util_alloc_full_path(path,table_config[ik]->eclipse_file);
    ecl_file_append = relperm_config_check_ecl_file(config->ecl_file_hash,table_config[ik]->eclipse_file,table_config[ik]->relptab_kw);
    if(ecl_file_append){
      relp_ecl_stream =enkf_util_fopen_a(relpfile,__func__);      
    }
    else{
      relp_ecl_stream =enkf_util_fopen_w(relpfile,__func__);
    }
    

    switch(table_config[ik]->func){
      
    case(COREY):
      {
	relperm_config_check_data(table_config[ik]->tab_corey,config->index_hash,data);

	switch(table_config[ik]->relptab_kw){
	case(SWOF):
	  {
	    relperm_config_ecl_write_corey_swof(relp_ecl_stream,table_config[ik]->tab_corey,config->index_hash, data, config->nsw,ecl_file_append);	    
	    break;
	  }
	case(SLGOF):
	  {
	    relperm_config_ecl_write_corey_slgof(relp_ecl_stream,table_config[ik]->tab_corey,config->index_hash, data);	    
	    break;
	  }
	default:
	  fprintf(stderr,"%s: This relperm table is not implemented",__func__);
	}

	break;
      }
    default:
      fprintf(stderr,"%s: This relperm function is not implemented",__func__);
      abort();
    }
    fclose(relp_ecl_stream);
    free(relpfile);
  }
  for(ik =0; ik < config->num_tab;ik++){free(table_config[ik]);}
  free(table_config);
}

void relperm_config_ecl_write_corey_swof(FILE * relp_ecl_stream, const table_corey_type * tab_corey, const hash_type * index_hash, const double * data, int nsw, bool ecl_file_append){
  /* The SWOF keyword may be used in runs containing both oil and water as active phases, to input 
     tables of water relative permeability. The table consists of 4 columns of data. 
     Column 1: The water saturation, 
     Column 2: The corresponding water relative permeability
     Column 3: The corresponding oil relative permeability when only oil and water are present, 
     Column 4: The corresponding water-oil capillary pressure 
  */

  double swco,soco,ewat,scwa,scoi,eowa,pewa;
  int i;
  double * swof1 = enkf_util_malloc(nsw * sizeof *swof1,__func__);
  double * swof2 = enkf_util_malloc(nsw * sizeof *swof2,__func__);
  double * swof3 = enkf_util_malloc(nsw * sizeof *swof3,__func__);
  double * swof4 = enkf_util_malloc(nsw * sizeof *swof4,__func__);

  swco = data[hash_get_int(index_hash,tab_corey->swco)];
  soco = data[hash_get_int(index_hash,tab_corey->soco)];
  ewat = data[hash_get_int(index_hash,tab_corey->ewat)];
  scwa = data[hash_get_int(index_hash,tab_corey->scwa)];
  scoi = data[hash_get_int(index_hash,tab_corey->scoi)];
  eowa = data[hash_get_int(index_hash,tab_corey->eowa)];
  pewa = data[hash_get_int(index_hash,tab_corey->pewa)];

  if(!ecl_file_append){
    fprintf(relp_ecl_stream,"%s \n","SWOF");
  }

  /*  fprintf(relp_ecl_stream,"swco: %g, soco:%g, ewat:%g, scwa:%g,scoi:%g,eowa:%g,pewa:%g \n",swco,soco,ewat,scwa,scoi,eowa,pewa);*/

  for(i=0;i < nsw ; i++){
    swof1[i] =  swco + ((1-swco)/(nsw-1))*i;
    swof2[i] = (swof1[i]-swco)/(1-swco-soco);
    if(swof2[i]<0) {swof2[i]=0.0;}
    if(swof2[i]>1) {swof2[i]=1.0;}
    swof2[i]=scwa*pow(swof2[i],ewat);
    swof3[i]=(1-swof1[i]-soco)/(1-swco-soco);
    if(swof3[i]<0){swof3[i]=0.0;}
    swof3[i]=scoi*pow(swof3[i],eowa);
    swof4[i]=exp(-pewa*(swof1[i]-swco));
    if(swof4[i]<0){swof4[i]=0.0;}
    fprintf(relp_ecl_stream,"%10.7f %10.7f %10.7f %10.7f \n",swof1[i],swof2[i],swof3[i],swof4[i]);
  }
  fprintf(relp_ecl_stream,"%s \n","/");
  free(swof1);
  free(swof2);
  free(swof3);
  free(swof4);
  
}
void relperm_config_ecl_write_corey_sgof(FILE * relp_ecl_stream, const table_corey_type * tab_corey, const hash_type * index_hash, const double * data, int nsw){
  fprintf(stderr,"%s: Not yet implemented",__func__);
  abort();
}
void relperm_config_ecl_write_corey_slgof(FILE * relp_ecl_stream, const table_corey_type * tab_corey, const hash_type * index_hash, const double * data){
  /* The SLGOF keyword may be used in runs containing both oil and gas as active phase, to input tables of 
     gas relperm, oil-in-gas relperm and oil-gas capillary pressure as a function of the liquid saturation.
     Column 1: The liquid saturation
     Column 2: The corresponding gas relperm
     Column 3: The corresponding oil relperm when oil, agas and connate water are present
     Column 4: The corresponding oil-gas capillary pressure
   */
  int index;
  double slgof;
  index = hash_get_int(index_hash,tab_corey->swco);
  slgof = data[index];
  fprintf(relp_ecl_stream,"SLGOF keyword test \n %s %g \n",tab_corey->swco,slgof);
}
void relperm_config_ecl_write_corey_swfn(FILE * relp_ecl_stream, const table_corey_type * tab_corey, const hash_type * index_hash, const double * data){
 fprintf(stderr,"%s: Not yet implemented",__func__);
 abort();
}
void relperm_config_ecl_write_corey_sgfn(FILE * relp_ecl_stream, const table_corey_type * tab_corey, const hash_type * index_hash, const double * data){
 fprintf(stderr,"%s: Not yet implemented",__func__);
 abort();
}
void relperm_config_ecl_write_corey_sof2(FILE * relp_ecl_stream, const table_corey_type * tab_corey, const hash_type * index_hash, const double * data){
 fprintf(stderr,"%s: Not yet implemented",__func__);
 abort();
}
void relperm_config_ecl_write_corey_sof3(FILE * relp_ecl_stream, const table_corey_type * tab_corey, const hash_type * index_hash, const double * data){
 fprintf(stderr,"%s: Not yet implemented",__func__);
 abort();
}
void relperm_config_ecl_write_corey_sof32d(FILE * relp_ecl_stream, const table_corey_type * tab_corey, const hash_type * index_hash, const double * data){
 fprintf(stderr,"%s: Not yet implemented",__func__);
 abort();
}


relptab_kw_type relperm_config_set_relptab_kw(char * relptab_kw_name){
  relptab_kw_type relptab_kw;
  if(strcmp(relptab_kw_name, "SWOF") == 0){
    relptab_kw = SWOF;
  }
  else if(strcmp(relptab_kw_name, "SGOF") == 0){
    relptab_kw = SGOF;
  }
  else if(strcmp(relptab_kw_name, "SLGOF") == 0){
    relptab_kw = SLGOF;
  }
  else if(strcmp(relptab_kw_name, "SWFN") == 0){
    relptab_kw = SWFN;
  }
  else if(strcmp(relptab_kw_name, "SGFN") == 0){
    relptab_kw = SGFN;
  }
  else if(strcmp(relptab_kw_name, "SOF2") == 0){
    relptab_kw = SOF2;
  }
  else if(strcmp(relptab_kw_name, "SOF3") == 0){
    relptab_kw = SOF3;
  }
  else if(strcmp(relptab_kw_name, "SOF32D") == 0){
    relptab_kw = SOF32D;
  }
  else{
    fprintf(stderr,"%s: The relperm keyword: %s is not valid",__func__,relptab_kw_name);
    abort();
  }
  return relptab_kw;
}

void relperm_config_check_data(const table_corey_type * tab_corey, const hash_type * index_hash, const double * data){
  double swco,soco,sgco,sorg;
  double ewat,egas,eowa,eogw;
  double scwa,scga,scoi;
  double pewa,pega;
  
  swco = data[hash_get_int(index_hash,tab_corey->swco)];
  soco = data[hash_get_int(index_hash,tab_corey->soco)];
  sgco = data[hash_get_int(index_hash,tab_corey->sgco)]; 
  sorg = data[hash_get_int(index_hash,tab_corey->sorg)]; 
  ewat = data[hash_get_int(index_hash,tab_corey->ewat)];
  egas = data[hash_get_int(index_hash,tab_corey->egas)];
  eowa = data[hash_get_int(index_hash,tab_corey->eowa)];
  eogw = data[hash_get_int(index_hash,tab_corey->eogw)];
  scwa = data[hash_get_int(index_hash,tab_corey->scwa)];
  scga = data[hash_get_int(index_hash,tab_corey->scga)];
  scoi = data[hash_get_int(index_hash,tab_corey->scoi)];
  pewa = data[hash_get_int(index_hash,tab_corey->pewa)];
  pega = data[hash_get_int(index_hash,tab_corey->pega)];

  /* Sanity check on connate levels */
  if( swco < 0 || swco > 1){
    fprintf(stderr,"%s: Error: swco must be between 0 and 1. swco = %g \n",__func__,swco);
    abort();
  }
  else if(soco < 0 || soco > 1){
    fprintf(stderr,"%s: Error: soco must be between 0 and 1. soco = %g \n",__func__,soco);
    abort();
  }
  else if(sgco < 0 || sgco > 1){
    fprintf(stderr,"%s: Error: sgco must be between 0 and 1. sgco = %g \n",__func__,sgco);
    abort();
  }
  else if(sorg < 0 || sorg > 1){
    fprintf(stderr,"%s: Error: sorg must be between 0 and 1. sorg = %g \n",__func__,sorg);
    abort();
  }
  else if(swco + soco + sgco > 1){
    fprintf(stderr,"%s: Error: The sum of connate levels must be less than 1. swco+soco+sgco = %g \n",__func__,swco+soco+sgco);
    abort();
  }
  else if(sgco > soco){
    fprintf(stderr,"%s: Warning: sgco > soco, this may cause convergence problems in the Brooks-Corey model",__func__);
  }
   
  /* Sanity check on exponents*/
  if(ewat < 0 ){
    fprintf(stderr,"%s: Error: ewat must be positive. ewat = %g \n",__func__,ewat);
    abort();
  }
  else if(egas < 0 ){
    fprintf(stderr,"%s: Error: egas must be positive. egas = %g \n",__func__,egas);
    abort();
  }
  else if(eowa < 0 ){
    fprintf(stderr,"%s: Error: eowa must be positive. eowa = %g \n",__func__,eowa);
    abort();
  }
  else if(eogw < 0 ){
    fprintf(stderr,"%s: Error: eogw must be positive. eogw = %g \n",__func__,eogw);
    abort();
  }
  
  /* Sanity check on scales */
  if(scwa < 0  || scwa > 1) {
    fprintf(stderr,"%s: Warning: scwa must be between 0 and 1. scwa = %g \n",__func__,scwa);
    abort();
  }
  else if(scga < 0  || scga > 1) {
    fprintf(stderr,"%s: Warning: scga must be between 0 and 1. scga = %g \n",__func__,scga);
    abort();
  }
  else if(scoi < 0  || scoi > 1) {
    fprintf(stderr,"%s: Warning: scoi must be between 0 and 1. scga = %g \n",__func__,scoi);
    abort();
  }
  
  /* Sanity check on pressure */
  if(pega < 0 ) {
    fprintf(stderr,"%s: Error: pega must be positive. pega = %g \n",__func__,pega);
    abort();
  }
  else if(pewa < 0 ) {
    fprintf(stderr,"%s: Error: pewa must be positive. pewa = %g \n",__func__,pewa);
    abort();
  }    
}

bool relperm_config_check_ecl_file(hash_type * ecl_file_hash, char * eclipse_file, relptab_kw_type relptab_kw){
    /* Check here if the eclipse file name in relperm_table.txt is used more than once.
       If so the following relperm tables that use the same eclipse file name will be appended 
       to the existing file. This requires that the table_config[ik]->relptab_kw is equal for the 
       different tables. 
       OK:      RELPTAB1 SWOF .....       ERROR:  RELPTAB1 SWOF  ......             
                RELPTAB1 SWOF .....               RELPTAB1 SLGOF ......
    */
  bool ecl_file_append;
  if(!hash_has_key(ecl_file_hash,eclipse_file)){
    hash_insert_int(ecl_file_hash,eclipse_file,relptab_kw);
    ecl_file_append = false;
  }
  else{
    if(hash_get_int(ecl_file_hash,eclipse_file)==relptab_kw){
    ecl_file_append = true;
    }
    else{
      fprintf(stderr,"Problem in %s. It is required that the relptab_kw's (SWOF, SLGOF, ..etc) have to be equal, when a specific eclipse file name is used more than once in relperm_table.txt",__func__);
      abort();	
    }
  }
  return ecl_file_append;
}
