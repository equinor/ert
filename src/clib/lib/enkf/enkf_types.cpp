#include <stdlib.h>

#include <ert/util/util.h>

#include <ert/enkf/enkf_types.hpp>

const char *enkf_types_get_impl_name(ert_impl_type impl_type) {
    switch (impl_type) {
    case (INVALID):
        return "INVALID";
    case FIELD:
        return "FIELD";
    case GEN_KW:
        return "GEN_KW";
    case SUMMARY:
        return "SUMMARY";
    case GEN_DATA:
        return "GEN_DATA";
    case EXT_PARAM:
        return "EXT_PARAM";
    default:
        util_abort("%s: internal error - unrecognized implementation type: %d "
                   "- aborting \n",
                   __func__, impl_type);
        return NULL;
    }
}
