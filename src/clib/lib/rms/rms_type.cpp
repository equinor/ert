#include <ert/rms/rms_type.hpp>
#include <stdlib.h>

/* A microscopic (purely internal) type object only used
   for storing the hash type_map */

void rms_type_free(void *rms_t) { free((__rms_type *)rms_t); }

static __rms_type *rms_type_set(__rms_type *rms_t, rms_type_enum rms_type,
                                int sizeof_ctype) {
    rms_t->rms_type = rms_type;
    rms_t->sizeof_ctype = sizeof_ctype;
    return rms_t;
}

__rms_type *rms_type_alloc(rms_type_enum rms_type, int sizeof_ctype) {
    __rms_type *rms_t = (__rms_type *)malloc(sizeof *rms_t);
    rms_type_set(rms_t, rms_type, sizeof_ctype);
    return rms_t;
}
