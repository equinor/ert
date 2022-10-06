#include <filesystem>

#include <stdlib.h>

#include <ert/util/util.hpp>

#include <ert/config/conf_data.hpp>

namespace fs = std::filesystem;

#define DT_STR_STRING "string"
#define DT_INT_STRING "integer"
#define DT_POSINT_STRING "positive integer"
#define DT_FLOAT_STRING "floating point number"
#define DT_POSFLOAT_STRING "positive floating foint number"
#define DT_FILE_STRING "file"
#define DT_DATE_STRING "date"

const char *conf_data_get_dt_name_ref(dt_enum dt) {
    switch (dt) {
    case (DT_STR):
        return DT_STR_STRING;
    case (DT_INT):
        return DT_INT_STRING;
    case (DT_POSINT):
        return DT_POSINT_STRING;
    case (DT_FLOAT):
        return DT_FLOAT_STRING;
    case (DT_POSFLOAT):
        return DT_POSFLOAT_STRING;
    case (DT_FILE):
        return DT_FILE_STRING;
    case (DT_DATE):
        return DT_DATE_STRING;
    default:
        util_abort("%s: Internal error.\n", __func__);
        return "";
    }
}

bool conf_data_validate_string_as_dt_value(dt_enum dt, const char *str) {
    if (str == NULL)
        return false;

    switch (dt) {
    case (DT_STR):
        return true;
    case (DT_INT):
        return util_sscanf_int(str, NULL);
    case (DT_POSINT): {
        int val;
        bool ok = util_sscanf_int(str, &val);
        if (!ok)
            return false;
        else
            return val > 0;
    }
    case (DT_FLOAT):
        return util_sscanf_double(str, NULL);
    case (DT_POSFLOAT): {
        double val;
        bool ok = util_sscanf_double(str, &val);
        if (!ok)
            return false;
        else
            return val >= 0.0;
    }
    case (DT_FILE): {
        return fs::exists(str);
    }
    case (DT_DATE): {
        time_t date;
        if (util_sscanf_isodate(str, &date))
            return true;
        return util_sscanf_date_utc(str, &date);
    }
    default:
        util_abort("%s: Error parsing \"%s\".\n", __func__, str);
    }
    return true;
}

int conf_data_get_int_from_string(dt_enum dt, const char *str) {
    int value = 0;
    bool ok = true;

    switch (dt) {
    case (DT_INT):
        ok = util_sscanf_int(str, &value);
        break;
    case (DT_POSINT):
        ok = util_sscanf_int(str, &value);
        break;
    default:
        ok = false;
    }

    if (!ok)
        util_abort("%s: Can not get an int from \"%s\".\n", __func__, str);

    return value;
}

double conf_data_get_double_from_string(dt_enum dt, const char *str) {
    double value = 0;
    bool ok = true;

    switch (dt) {
    case (DT_INT):
        ok = util_sscanf_double(str, &value);
        break;
    case (DT_POSINT):
        ok = util_sscanf_double(str, &value);
        break;
    case (DT_FLOAT):
        ok = util_sscanf_double(str, &value);
        break;
    case (DT_POSFLOAT):
        ok = util_sscanf_double(str, &value);
        break;
    default:
        ok = false;
    }

    if (!ok)
        util_abort("%s: Can not get a double from \"%s\".\n", __func__, str);

    return value;
}

time_t conf_data_get_time_t_from_string(dt_enum dt, const char *str) {
    time_t value = 0;
    bool ok = true;

    switch (dt) {
    case (DT_DATE):
        ok = util_sscanf_isodate(str, &value);
        if (!ok) {
            ok = util_sscanf_date_utc(str, &value);
        }
        break;
    default:
        ok = false;
    }

    if (!ok)
        util_abort("%s: Can not get a time_t from \"%s\".\n", __func__, str);
    return value;
}
