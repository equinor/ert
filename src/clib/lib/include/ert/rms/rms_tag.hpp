#ifndef ERT_RMS_TAG_H
#define ERT_RMS_TAG_H
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include <ert/util/hash.hpp>

#include <ert/rms/rms_tagkey.hpp>

typedef struct rms_tag_struct rms_tag_type;

const char *rms_tag_get_namekey_name(const rms_tag_type *);
const char *rms_tag_get_name(const rms_tag_type *);
void rms_tag_free(rms_tag_type *);
void rms_tag_free__(void *arg);
rms_tag_type *rms_tag_fread_alloc(FILE *, hash_type *, bool, bool *);
bool rms_tag_name_eq(const rms_tag_type *, const char *, const char *,
                     const char *);
rms_tagkey_type *rms_tag_get_key(const rms_tag_type *, const char *);
void rms_tag_fwrite_filedata(const char *, FILE *stream);
void rms_tag_fwrite_eof(FILE *stream);
void rms_tag_fwrite(const rms_tag_type *, FILE *);
rms_tag_type *rms_tag_alloc_dimensions(int, int, int);
void rms_tag_fwrite_dimensions(int, int, int, FILE *);
void rms_tag_fwrite_parameter(const char *, const rms_tagkey_type *, FILE *);
#endif
