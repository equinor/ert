#ifndef __DATA_TYPE_H__
#define __DATA_TYPE_H__
#include <stdbool.h>

typedef enum {
              DATA_TYPE_STR,
              DATA_TYPE_INT,
              DATA_TYPE_POSINT,
              DATA_TYPE_FLOAT,
              DATA_TYPE_POSFLOAT,
              DATA_TYPE_FILE,
              DATA_TYPE_DATE
              } data_type_enum;


data_type_enum get_data_type_from_string(const char *);
bool is_data_type(const char *);
#endif
