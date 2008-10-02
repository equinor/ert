#ifndef __MODEL_CONFIG_H__
#define __MODEL_CONFIG_H__
#include <time.h>
#include <config.h>

typedef struct model_config_struct model_config_type;

model_config_type * model_config_alloc(const config_type *  , time_t );
void                model_config_free(model_config_type *);

#endif
