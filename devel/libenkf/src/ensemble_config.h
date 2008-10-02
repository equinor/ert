#ifndef __ENSEMBLE_CONFIG_H__
#define __ENSEMBLE_CONFIG_H__



typedef struct ensemble_config_struct ensemble_config_type;


ensemble_config_type * ensemble_config_alloc(int );
void                   ensemble_config_free(ensemble_config_type * );


#endif
