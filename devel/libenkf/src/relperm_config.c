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
  relperm_config->kw_list       = util_malloc(size_conf * sizeof  *relperm_config->kw_list, __func__);
  relperm_config->index_hash    = hash_alloc();
  relperm_config->ecl_file_hash = hash_alloc();
  
  relperm_config->table_config  = util_malloc(size_tab * sizeof *relperm_config->table_config,__func__); 
  
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
    config->table_config[line_nr]=relperm_config_fscanf_table_config_alloc(stream_table,config->ecl_file_hash);
    relperm_config_check_tab_input(config->index_hash,config->table_config[line_nr],config->famnr);
    printf("config->table_config->eclipse_file %s \n",config->table_config[line_nr]->eclipse_file);
    line_nr++;
  }while(line_nr < size_tab-1);  

  fclose(stream_table);
  fclose(stream_config);

  return config;
}

table_config_type * relperm_config_fscanf_table_config_alloc(FILE * stream_tab, hash_type * ecl_file_hash ){
  char ** token_list;
  int tokens;
  char * line;
  bool at_eof = false;

  table_config_type * tab_config = util_malloc(sizeof *tab_config,__func__);      

  line = util_fscanf_alloc_line(stream_tab,&at_eof);
  if (line != NULL){
    util_split_string(line," ", &tokens, &token_list);
    /* tokens has to be equal to 16 or else the line in relperm_table.txt is not correctly specified */ 
    if (tokens == 16){
      tab_config->eclipse_file = util_alloc_string_copy(token_list[0]);
      tab_config->relptab_kw   = relperm_config_set_relptab_kw(token_list[1]);
      tab_config->ecl_file_append = relperm_config_check_ecl_file(ecl_file_hash,tab_config->eclipse_file,tab_config->relptab_kw);
    } else
      util_abort("%s: The line in relperm_table.txt is not correctly specified.\n The number of arguments should be 16, and the line has %d arguments",__func__,tokens);
  }
  else 
    util_abort("%s: Something wrong when reading relperm_tab.txt",__func__);

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
      tab_config->tab = relperm_config_table_alloc(token_list,tokens);
      break;
    }  
  default:
    fprintf(stderr,"Problem in %s. Wrong input in relperm_tab.txt: Options are: COREY \n",__func__);
    abort();
  }

  util_free_stringlist(token_list,tokens);
  free(line);
  return tab_config;
}

table_type * relperm_config_table_alloc(char ** token_list, int tokens){
    table_type * tab = util_malloc(sizeof *tab,__func__);      
    tab->swco = util_alloc_string_copy(token_list[3]);
    tab->soco = util_alloc_string_copy(token_list[4]);
    tab->sgco = util_alloc_string_copy(token_list[5]);
    tab->sorg = util_alloc_string_copy(token_list[6]);
    tab->ewat = util_alloc_string_copy(token_list[7]);
    tab->egas = util_alloc_string_copy(token_list[8]);
    tab->eowa = util_alloc_string_copy(token_list[9]);
    tab->eogw = util_alloc_string_copy(token_list[10]);
    tab->scwa = util_alloc_string_copy(token_list[11]);
    tab->scga = util_alloc_string_copy(token_list[12]);
    tab->scoi = util_alloc_string_copy(token_list[13]);
    tab->pewa = util_alloc_string_copy(token_list[14]);
    tab->pega = util_alloc_string_copy(token_list[15]);
    return tab;
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
      table_type * tab = tab_config->tab;
      if(!hash_has_key(index_hash,tab->swco)){
        fprintf(stderr,"%s: The keyword %s for tab->swco is not defined in relperm_config.txt \n",__func__,tab->swco);
	abort();
      }
      else if(!hash_has_key(index_hash,tab->soco)){
	fprintf(stderr,"%s: The keyword %s for tab->soco is not defined in relperm_config.txt \n",__func__,tab->soco);
	abort();
      }
      else if(!hash_has_key(index_hash,tab->sgco)){
	fprintf(stderr,"%s: The keyword %s for tab->sgco is not defined in relperm_config.txt \n",__func__,tab->sgco);
	abort();
      }
      else if(!hash_has_key(index_hash,tab->sorg)){
	fprintf(stderr,"%s: The keyword %s for tab->sorg is not defined in relperm_config.txt \n",__func__,tab->sorg);
	abort();
      }
      else if(!hash_has_key(index_hash,tab->ewat)){
	fprintf(stderr,"%s: The keyword %s for tab->ewat is not defined in relperm_config.txt \n",__func__,tab->ewat);
	abort();
      }
      else if(!hash_has_key(index_hash,tab->egas)){
	fprintf(stderr,"%s: The keyword %s for tab->egas is not defined in relperm_config.txt \n",__func__,tab->egas);
	abort();
      }
      else if(!hash_has_key(index_hash,tab->eowa)){
	fprintf(stderr,"%s: The keyword %s for tab->eowa is not defined in relperm_config.txt \n",__func__,tab->eowa);
	abort();
      }
      else if(!hash_has_key(index_hash,tab->eogw)){
	fprintf(stderr,"%s: The keyword %s for tab->eogw is not defined in relperm_config.txt \n",__func__,tab->eogw);
	abort();
      }
      else if(!hash_has_key(index_hash,tab->scwa)){
	fprintf(stderr,"%s: The keyword %s for tab->scwa is not defined in relperm_config.txt \n",__func__,tab->scwa);
	abort();
      }
      else if(!hash_has_key(index_hash,tab->scga)){
	fprintf(stderr,"%s: The keyword %s for tab->scga is not defined in relperm_config.txt \n",__func__,tab->scga);
	abort();
      }
      else if(!hash_has_key(index_hash,tab->scoi)){
	fprintf(stderr,"%s: The keyword %s for tab->scoi is not defined in relperm_config.txt \n",__func__,tab->scoi);
	abort();
      }
      else if(!hash_has_key(index_hash,tab->pewa)){
	fprintf(stderr,"%s: The keyword %s for tab->pewa is not defined in relperm_config.txt \n",__func__,tab->pewa);
	abort();
      }
      else if(!hash_has_key(index_hash,tab->pega)){
	fprintf(stderr,"%s: The keyword %s for tab->pega is not defined in relperm_config.txt \n",__func__,tab->pega);
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

void relperm_config_ecl_write(const relperm_config_type * relperm_config,const double * data,  FILE * stream,char * eclpath){
  const char * kw;
  kw = hash_iter_get_first_key(relperm_config->ecl_file_hash);
  while(kw !=NULL){
    
    fprintf(stream,"INCLUDE \n %s / \n",kw);
    kw=hash_iter_get_next_key(relperm_config->ecl_file_hash);
  }
  relperm_config_ecl_write_table(relperm_config, data, eclpath);
}

void relperm_config_ecl_write_table(const relperm_config_type * config, const double * data, const char * path){
  int ik;
  table_config_type ** table_config;
  table_config = config->table_config;
  
  for(ik =0; ik < config->num_tab;ik++){
    FILE * relp_ecl_stream;
    char * relpfile;
    
    relpfile = util_alloc_full_path(path,table_config[ik]->eclipse_file);
    /* ecl_file_append = relperm_config_check_ecl_file(config->ecl_file_hash,table_config[ik]->eclipse_file,table_config[ik]->relptab_kw);*/
    if(table_config[ik]->ecl_file_append){
      relp_ecl_stream = util_fopen(relpfile, "a");      
    }
    else{
      relp_ecl_stream =util_fopen(relpfile, "r");
    }
    
    relperm_config_check_data(table_config[ik]->tab,config->index_hash,data,config->famnr);

    switch(table_config[ik]->relptab_kw){
    case(SWOF):
      {
	relperm_config_ecl_write_swof(relp_ecl_stream,table_config[ik]->tab,config->index_hash, data, config->nsw,table_config[ik]->ecl_file_append,table_config[ik]->func);	    
	break;
      }
    case(SLGOF):
      {
	relperm_config_ecl_write_slgof(relp_ecl_stream,table_config[ik]->tab,config->index_hash, data,config->nso,table_config[ik]->ecl_file_append,table_config[ik]->func);	    
	break;
      }
    case(SGOF):
      {
	relperm_config_ecl_write_sgof(relp_ecl_stream,table_config[ik]->tab,config->index_hash, data,config->nso,table_config[ik]->ecl_file_append,table_config[ik]->func);
	break;
      }
    case(SWFN):
      {
	relperm_config_ecl_write_swfn(relp_ecl_stream,table_config[ik]->tab,config->index_hash, data,config->nsw,table_config[ik]->ecl_file_append,table_config[ik]->func);	    
	break;
      }
    case(SGFN):
      {
	relperm_config_ecl_write_sgfn(relp_ecl_stream,table_config[ik]->tab,config->index_hash, data,config->nsg,table_config[ik]->ecl_file_append,table_config[ik]->func);	    
	break;
      }
    case(SOF3):
      {
	relperm_config_ecl_write_sof3(relp_ecl_stream,table_config[ik]->tab,config->index_hash, data,config->nso,table_config[ik]->ecl_file_append,table_config[ik]->func);	    
	break;
      }
    default:
      {
      fprintf(stderr,"%s: This relperm table is not implemented",__func__);
      }
    }
    fclose(relp_ecl_stream);
    free(relpfile);
  }
}


void relperm_config_ecl_write_swof(FILE * relp_ecl_stream, const table_type * tab, const hash_type * index_hash, const double * data, int nsw, bool ecl_file_append, func_type func){
  /* The SWOF keyword may be used in runs containing both oil and water as active phases, to input 
     tables of water relative permeability. The table consists of 4 columns of data. 
     Column 1: The water saturation, 
     Column 2: The corresponding water relative permeability
     Column 3: The corresponding oil relative permeability when only oil and water are present, 
     Column 4: The corresponding water-oil capillary pressure 
  */

  double swco,soco,ewat,scwa,scoi,eowa,pewa;
  int i;
  double * swof1 = util_malloc(nsw * sizeof *swof1,__func__);
  double * swof2 = util_malloc(nsw * sizeof *swof2,__func__);
  double * swof3 = util_malloc(nsw * sizeof *swof3,__func__);
  double * swof4 = util_malloc(nsw * sizeof *swof4,__func__);
  
  swco = data[hash_get_int(index_hash,tab->swco)];
  soco = data[hash_get_int(index_hash,tab->soco)];
  ewat = data[hash_get_int(index_hash,tab->ewat)];
  scwa = data[hash_get_int(index_hash,tab->scwa)];
  scoi = data[hash_get_int(index_hash,tab->scoi)];
  eowa = data[hash_get_int(index_hash,tab->eowa)];
  pewa = data[hash_get_int(index_hash,tab->pewa)];

  fprintf(relp_ecl_stream,"--swco: %g \n--soco: %g \n--ewat: %g \n--scwa: %g \n--scoi: %g \n--eowa: %g \n--pewa: %g \n",swco,soco,ewat,scwa,scoi,eowa,pewa);

  if(!ecl_file_append){
    fprintf(relp_ecl_stream,"%s \n","SWOF");
  }
  if(func == COREY){
    for(i=0;i < nsw ; i++){
      /*      swof1[i] =  swco + ((1-swco)/(nsw-1))*i; Should be used for drainage*/
      swof1[i] =  swco + ((1-swco-soco)/(nsw-1))*i;
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
  }
  fprintf(relp_ecl_stream,"%s \n","/");

  free(swof1);
  free(swof2);
  free(swof3);
  free(swof4);  
}

void relperm_config_ecl_write_sgof(FILE * relp_ecl_stream, const table_type * tab, const hash_type * index_hash, const double * data, int nso, bool ecl_file_append, func_type func){
  /* The SGOF keyword may be used in runs containing both oil and gas as active phases, to input tables of gas relperm, 
     oil-in-gas relperm and oil-gas and oil-gas capillary pressure as function of gas saturation.
     Column 1: The gas saturation
     Column 2: The corresponding gas relperm
     Column 3: The corrsponding oil relperm when oil, gas and connate water are present
     Column 4: The corresponding oil-gas cappilary pressure
   */
  double swco,soco,sgco,scga,scoi,egas,eogw;
  int i;
  
  double * sgof1 = util_malloc(nso * sizeof *sgof1,__func__);
  double * sgof2 = util_malloc(nso * sizeof *sgof2,__func__);
  double * sgof3 = util_malloc(nso * sizeof *sgof3,__func__);
  double * sgof4 = util_malloc(nso * sizeof *sgof4,__func__);

  swco = data[hash_get_int(index_hash,tab->swco)];
  soco = data[hash_get_int(index_hash,tab->soco)];
  sgco = data[hash_get_int(index_hash,tab->sgco)];
  scga = data[hash_get_int(index_hash,tab->scga)];
  scoi = data[hash_get_int(index_hash,tab->scoi)];
  egas = data[hash_get_int(index_hash,tab->egas)];
  eogw = data[hash_get_int(index_hash,tab->eogw)];
  
  fprintf(relp_ecl_stream,"--swco: %g\n--soco: %g\n--sgco: %g\n--scga: %g\n--scoi: %g\n--egas: %g\n--eogw: %g \n",swco,soco,sgco,scga,scoi,egas,eogw);

  if(!ecl_file_append){
    fprintf(relp_ecl_stream, "%s \n", "SGOF");
  }
  if(func == COREY){
    for(i=0;i <nso;i++){
      sgof1[i]= 0.0 +((1.0-0.0)/(nso-1))*i;
      printf("sgof1[i] =%g",sgof1[i]);
      sgof2[i]= (sgof1[i]-sgco)/(1-sgco-swco-soco);
      if(sgof2[i] < 0){sgof2[i] = 0.0;}
      if(sgof2[i] > 1){sgof2[i] = 1.0;}
      sgof2[i]=scga*pow(sgof2[i],egas);
      sgof3[i]=(1-sgof1[i]-swco-soco)/(1-swco-soco);
      if(sgof3[i] < 0){sgof3[i] = 0.0;}
      sgof3[i]=scoi*pow(sgof3[i],eogw);
      sgof4[i]=0.0;
      fprintf(relp_ecl_stream,"%10.7f %10.7f %10.7f %10.7f \n",sgof1[i],sgof2[i],sgof3[i],sgof4[i]);
    }
  }
  fprintf(relp_ecl_stream,"%s \n","/");
  free(sgof1);
  free(sgof2);
  free(sgof3);
  free(sgof4);
}

void relperm_config_ecl_write_slgof(FILE * relp_ecl_stream, const table_type * tab, const hash_type * index_hash, const double * data,int nso, bool ecl_file_append, func_type func ){
  /* The SLGOF keyword may be used in runs containing both oil and gas as active phase, to input tables of 
     gas relperm, oil-in-gas relperm and oil-gas capillary pressure as a function of the liquid saturation.
     Column 1: The liquid saturation
     Column 2: The corresponding gas relperm
     Column 3: The corresponding oil relperm when oil, agas and connate water are present
     Column 4: The corresponding oil-gas capillary pressure
   */
  double swco,soco,sgco,sorg, scga, scoi,egas,eogw,pega;
  int i;
  double * slgof1 = util_malloc(nso * sizeof *slgof1,__func__);
  double * slgof2 = util_malloc(nso * sizeof *slgof2,__func__);
  double * slgof3 = util_malloc(nso * sizeof *slgof3,__func__);
  double * slgof4 = util_malloc(nso * sizeof *slgof4,__func__);

  swco = data[hash_get_int(index_hash,tab->swco)];
  soco = data[hash_get_int(index_hash,tab->soco)];
  sgco = data[hash_get_int(index_hash,tab->sgco)];
  sorg = data[hash_get_int(index_hash,tab->sorg)];
  scga = data[hash_get_int(index_hash,tab->scga)];
  scoi = data[hash_get_int(index_hash,tab->scoi)];
  egas = data[hash_get_int(index_hash,tab->egas)];
  eogw = data[hash_get_int(index_hash,tab->eogw)];
  pega = data[hash_get_int(index_hash,tab->pega)];

  fprintf(relp_ecl_stream,"--swco: %g\n--soco: %g\n--sgco: %g\n--sorg: %g\n--scga: %g\n--scoi: %g\n--egas: %g\n--eogw: %g \n",swco,soco,sgco,sorg,scga,scoi,egas,eogw);

  if(!ecl_file_append){
    fprintf(relp_ecl_stream, "%s \n", "SLGOF");
  }
  if(func ==COREY){
    for(i=0;i < nso; i++){
      slgof1[i]=swco +soco + ((1-swco-soco)/(nso-1))*i;
      slgof2[i]=(1-slgof1[i]-sgco)/(1-sgco-swco-soco);
      if(slgof2[i] < 0){slgof2[i] = 0.0;}
      if(slgof2[i] > 1){slgof2[i] = 1.0;}
      slgof2[i]=scga*pow(slgof2[i],egas);
      slgof3[i]=(slgof1[i]-swco-soco)/(1-swco-soco);
      slgof3[i]=scoi*pow(slgof3[i],eogw);
      slgof4[i]=exp(-pega*(slgof1[i]-sgco));
      if(slgof4[i]<0){(slgof4[i]=0.0);}
      fprintf(relp_ecl_stream,"%10.7f %10.7f %10.7f %10.7f \n",slgof1[i],slgof2[i],slgof3[i],slgof4[i]);
    }
  }
  fprintf(relp_ecl_stream,"%s \n","/");
  free(slgof1);
  free(slgof2);
  free(slgof3);
  free(slgof4);
}
void relperm_config_ecl_write_swfn(FILE * relp_ecl_stream, const table_type * tab, const hash_type * index_hash, const double * data,int nsw, bool ecl_file_append, func_type func){
  /* Water saturation functions 
     Column 1: The water saturation
     Column 2: The corresponding water relperm
     Column 3: The corresponding water-oil capillary pressure
  */
  double swco,soco,scwa,ewat;
  int i;
  double * swfn1 = util_malloc(nsw * sizeof *swfn1,__func__);
  double * swfn2 = util_malloc(nsw * sizeof *swfn2,__func__);
  double * swfn3 = util_malloc(nsw * sizeof *swfn3,__func__);
  
  swco = data[hash_get_int(index_hash,tab->swco)];
  soco = data[hash_get_int(index_hash,tab->soco)];
  scwa = data[hash_get_int(index_hash,tab->scwa)];
  ewat = data[hash_get_int(index_hash,tab->ewat)];

  fprintf(relp_ecl_stream,"--swco: %g \n--soco: %g \n--scwa: %g \n--ewat: %g\n",swco,soco,scwa,ewat);

  if(!ecl_file_append){
    fprintf(relp_ecl_stream,"%s \n","SWFN");
  }
  if(func == COREY){
    for(i=0;i<nsw; i++){ 
    swfn1[i]=swco + ((1-swco)/(nsw-1))*i;
    swfn2[i]=(swfn1[i]-swco)/(1-swco-soco);
    if(swfn2[i] <0){swfn2[i] = 0.0;}
    if(swfn2[i] >1){swfn2[i] = 1.0;}
    swfn2[i]=scwa*pow(swfn2[i],ewat);
    /* Calculate the capillary pressure */
    swfn3[i]= 0;
    fprintf(relp_ecl_stream,"%10.7f %10.7f %10.7f \n",swfn1[i],swfn2[i],swfn3[i]);
    }
  }
  fprintf(relp_ecl_stream,"%s \n","/");
  free(swfn1);
  free(swfn2);
  free(swfn3);
}

void relperm_config_ecl_write_sgfn(FILE * relp_ecl_stream, const table_type * tab, const hash_type * index_hash, const double * data,int nsg, bool ecl_file_append, func_type func){
  /* Gas saturation functions
     Column 1: The gas saturation
     Column 2: The corresponding gas relperm
     Column 3: The corresponding oil-gas capillary pressure
  */
  double sgco,swco,sorg,scga,egas;
  int i;

  double * sgfn1 = util_malloc(nsg * sizeof *sgfn1,__func__); 
  double * sgfn2 = util_malloc(nsg * sizeof *sgfn2,__func__); 
  double * sgfn3 = util_malloc(nsg * sizeof *sgfn3,__func__); 
  
  sgco = data[hash_get_int(index_hash,tab->sgco)];
  swco = data[hash_get_int(index_hash,tab->swco)];
  sorg = data[hash_get_int(index_hash,tab->sorg)];
  scga = data[hash_get_int(index_hash,tab->scga)];
  egas = data[hash_get_int(index_hash,tab->egas)];

  fprintf(relp_ecl_stream,"--swco: %g\n--sgco:%g\n--sorg: %g\n--scga: %g\n--egas:%g \n",swco,sgco,sorg,scga,egas);

  if(!ecl_file_append){
    fprintf(relp_ecl_stream,"%s \n","SGFN");
  }
  if(func == COREY){
    for(i =0; i<nsg; i++){
      sgfn1[i] = sgco + ((1-swco-sgco)/(nsg-1))*i;
      sgfn2[i] = (sgfn1[i]-sgco)/(1-sgco-sorg);
      if(sgfn2[i] < 0){sgfn2[i] = 0.0;}
      if(sgfn2[i] > 1){sgfn2[i] = 1.0;}
      sgfn2[i]=scga*pow(sgfn2[i],egas);
      /* Calculate the capillary pressure */
      sgfn3[i] = 0.0;
      fprintf(relp_ecl_stream,"%10.7f %10.7f %10.7f \n",sgfn1[i],sgfn2[i],sgfn3[i]);
    }
  }
  fprintf(relp_ecl_stream,"%s \n","/");
  free(sgfn1);
  free(sgfn2);
  free(sgfn3);
}
void relperm_config_ecl_write_sof2(FILE * relp_ecl_stream, const table_type * tab, const hash_type * index_hash, const double * data){
 fprintf(stderr,"%s: Not yet implemented",__func__);
 abort();
}
void relperm_config_ecl_write_sof3(FILE * relp_ecl_stream, const table_type * tab, const hash_type * index_hash, const double * data,int nso, bool ecl_file_append, func_type func){
  /* Oil saturation functions (three phase) 
     Column 1: The oil saturation
     Column 2: The corresponding oil relperm for regions where only oil and water are present
     Column 3: The corresponding oil relperm for regions where only oil, gas and connate water are present
  */

  double swco,soco,sgco,sorg,scoi,eowa,scga,eogw;
  int i;
  
  double * sof31 = util_malloc(nso * sizeof *sof31, __func__);
  double * sof32 = util_malloc(nso * sizeof *sof32, __func__);
  double * sof33 = util_malloc(nso * sizeof *sof33, __func__);

  swco = data[hash_get_int(index_hash,tab->swco)];
  soco = data[hash_get_int(index_hash,tab->soco)];
  sgco = data[hash_get_int(index_hash,tab->sgco)];
  sorg = data[hash_get_int(index_hash,tab->sorg)];
  scoi = data[hash_get_int(index_hash,tab->scoi)];
  eowa = data[hash_get_int(index_hash,tab->eowa)];
  scga = data[hash_get_int(index_hash,tab->scga)];
  eogw = data[hash_get_int(index_hash,tab->eogw)];
  
  fprintf(relp_ecl_stream,"--swco: %g\n--soco: %g\n--sgco:%g\n--sorg: %g\n--scoi: %g\n--scga: %g\n--eogw: %g \n",swco,soco,sgco,sorg,scoi,scga,eogw);

  if(!ecl_file_append){
    fprintf(relp_ecl_stream,"%s \n","SOF3");
  }
  /* soco should allways have a lower value than sorg. See Eclipe technical report chap 47 p 700 */
  if(sorg < soco){sorg = soco;}
  if(func == COREY){
    for(i = 0; i<nso; i++){
      sof31[i] = soco + ((1-soco-swco)/(nso-1))*i;
      sof32[i] = (sof31[i]-soco)/(1-soco-swco);
      sof33[i] = (sof31[i]-sorg)/(1-sgco-sorg);
      if(sof32[i] < 0){sof32[i] = 0.0;}
      if(sof33[i] < 0){sof33[i] = 0.0;}
      sof32[i] = scoi*pow(sof32[i],eowa);
      sof33[i] = scga*pow(sof33[i],eogw);
      fprintf(relp_ecl_stream,"%10.7f %10.7f %10.7f \n",sof31[i],sof32[i],sof33[i]);
    }
  }
  fprintf(relp_ecl_stream,"%s \n","/");
  free(sof31);
  free(sof32);
  free(sof33);
}
void relperm_config_ecl_write_sof32d(FILE * relp_ecl_stream, const table_type * tab, const hash_type * index_hash, const double * data){
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

void relperm_config_check_data(const table_type * tab, const hash_type * index_hash, const double * data, int famnr){
  double swco,soco,sgco,sorg;
  double ewat,egas,eowa,eogw;
  double scwa,scga,scoi;
  double pewa,pega;
  
  swco = data[hash_get_int(index_hash,tab->swco)];
  soco = data[hash_get_int(index_hash,tab->soco)];
  sgco = data[hash_get_int(index_hash,tab->sgco)]; 
  sorg = data[hash_get_int(index_hash,tab->sorg)]; 
  ewat = data[hash_get_int(index_hash,tab->ewat)];
  egas = data[hash_get_int(index_hash,tab->egas)];
  eowa = data[hash_get_int(index_hash,tab->eowa)];
  eogw = data[hash_get_int(index_hash,tab->eogw)];
  scwa = data[hash_get_int(index_hash,tab->scwa)];
  scga = data[hash_get_int(index_hash,tab->scga)];
  scoi = data[hash_get_int(index_hash,tab->scoi)];
  pewa = data[hash_get_int(index_hash,tab->pewa)];
  pega = data[hash_get_int(index_hash,tab->pega)];

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

  /*Sanity check on soco and sorg for family 2 */
  if(famnr == 2){
    if(sorg < soco){
      fprintf(stderr,"%s: Warning: sorg < soco, sorg=soco, because soco should be the highest value for both oil-water and oil-gas where relperms are zero. Done in relperm_config_ecl_write_sof3 \n",__func__);
    }
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
    /* This routine checks if the eclipse file name in relperm_table.txt is used more than once.
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
void relperm_config_free(relperm_config_type * relperm_config){
  util_free_stringlist(relperm_config->kw_list, scalar_config_get_data_size(relperm_config->scalar_config));
  free(relperm_config->index_hash);
  free(relperm_config->ecl_file_hash);
  scalar_config_free(relperm_config->scalar_config);
  relperm_config_table_config_free(relperm_config->table_config, relperm_config->num_tab);
  free(relperm_config);
}
void relperm_config_table_config_free(table_config_type ** table_config,int num_tab){
  int i;
  for(i=0; i<num_tab; i++){
    free(table_config[i]->eclipse_file);
    relperm_config_table_free(table_config[i]->tab);
  }
  free(table_config);
}
void relperm_config_table_free(table_type * tab){
  free(tab->swco);
  free(tab->soco);
  free(tab->sgco);
  free(tab->sorg);
  free(tab->ewat);
  free(tab->egas);
  free(tab->eowa);
  free(tab->scwa);
  free(tab->scga);
  free(tab->scoi);
  free(tab->pewa);
  free(tab->pega);
  free(tab);
}
/*****************************************************************/
VOID_FREE(relperm_config)
