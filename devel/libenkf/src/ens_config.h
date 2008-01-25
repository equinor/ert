#ifndef __ENS_CONFIG_H__
#define __ENS_CONFIG_H__


typedef struct ens_config_struct ens_config_type;

ens_config_type * ens_config_alloc(int , const char *);
char            * ens_config_alloc_ensname(const ens_config_type * , const char * );
void              ens_config_free(ens_config_type *);


#endif
