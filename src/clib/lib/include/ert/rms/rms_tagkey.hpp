#ifndef ERT_RMS_TAGKEY_H
#define ERT_RMS_TAGKEY_H

#include <stdbool.h>
#include <stdio.h>

#include <ert/util/hash.hpp>

#include <ert/rms/rms_type.hpp>

#include <ert/ecl/ecl_util.h>

typedef struct rms_tagkey_struct rms_tagkey_type;

void rms_tagkey_free(rms_tagkey_type *);
rms_tagkey_type *rms_tagkey_alloc_empty(bool);
rms_tagkey_type *rms_tagkey_alloc_complete(const char *, int, rms_type_enum,
                                           const void *, bool);
const char *rms_tagkey_get_name(const rms_tagkey_type *);
rms_type_enum rms_tagkey_get_rms_type(const rms_tagkey_type *);
ecl_data_type rms_tagkey_get_ecl_data_type(const rms_tagkey_type *);
void rms_tagkey_set_data(rms_tagkey_type *, const void *);

bool rms_tagkey_char_eq(const rms_tagkey_type *, const char *);
void rms_tagkey_free_(void *);
void rms_tagkey_load(rms_tagkey_type *, bool, FILE *, hash_type *);
void *rms_tagkey_get_data_ref(const rms_tagkey_type *);
void rms_tagkey_fwrite(const rms_tagkey_type *, FILE *);
rms_tagkey_type *rms_tagkey_copyc(const rms_tagkey_type *);
int rms_tagkey_get_size(const rms_tagkey_type *);

rms_tagkey_type *rms_tagkey_alloc_byteswap();
rms_tagkey_type *rms_tagkey_alloc_creationDate();
rms_tagkey_type *rms_tagkey_alloc_filetype(const char *);
rms_tagkey_type *rms_tagkey_alloc_dim(const char *, int);
rms_tagkey_type *rms_tagkey_alloc_parameter_name(const char *);

void rms_tagkey_inplace_log10(rms_tagkey_type *);
void rms_tagkey_inplace_sqr(rms_tagkey_type *);
void rms_tagkey_inplace_sqrt(rms_tagkey_type *);
void rms_tagkey_inplace_add_scaled(rms_tagkey_type *, const rms_tagkey_type *,
                                   double);
void rms_tagkey_clear(rms_tagkey_type *);

#endif
