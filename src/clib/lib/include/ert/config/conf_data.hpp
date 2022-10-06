#ifndef ERT_CONF_DATA_H
#define ERT_CONF_DATA_H

#include <stdbool.h>
#include <time.h>

typedef enum {
    DT_STR,
    DT_INT,
    DT_POSINT,
    DT_FLOAT,
    DT_POSFLOAT,
    DT_FILE,
    DT_EXEC,
    DT_FOLDER,
    DT_DATE
} dt_enum;

const char *conf_data_get_dt_name_ref(dt_enum dt);

bool conf_data_validate_string_as_dt_value(dt_enum dt, const char *str);

int conf_data_get_int_from_string(dt_enum dt, const char *str);

double conf_data_get_double_from_string(dt_enum dt, const char *str);

time_t conf_data_get_time_t_from_string(dt_enum dt, const char *str);

#endif
