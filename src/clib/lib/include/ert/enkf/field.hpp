#ifndef ERT_FIELD_H
#define ERT_FIELD_H
#include <ert/util/type_macros.h>

#include <ert/ecl/ecl_kw.h>
#include <ert/ecl/ecl_type.h>
#include <ert/ecl/fortio.h>

#include <ert/enkf/enkf_macros.hpp>
#include <ert/enkf/enkf_serialize.hpp>
#include <ert/enkf/enkf_util.hpp>
#include <ert/enkf/field_common.hpp>
#include <ert/enkf/field_config.hpp>

/* Typedef field_type moved to field_config.h */

void field_scale(field_type *field, double scale_factor);
extern "C" double field_iget_double(const field_type *, int);
extern "C" double field_ijk_get_double(const field_type *field, int, int, int);
void field_ecl_write1D_fortio(const field_type *, fortio_type *);
void field_ecl_write3D_fortio(const field_type *, fortio_type *, const char *);
void field_ROFF_export(const field_type *, const char *, const char *);
void field_copy_ecl_kw_data(field_type *, const ecl_kw_type *);
extern "C" void field_free(field_type *);
bool field_fload_keep_inactive(field_type *field, const char *filename);
bool field_fload_rms(field_type *field, const char *filename,
                     bool keep_inactive);
void field_export3D(const field_type *, void *, bool, ecl_data_type, void *,
                    const char *);
extern "C" void field_export(const field_type *, const char *, fortio_type *,
                             field_file_format_type, bool, const char *);
extern "C" int field_get_size(const field_type *field);

void field_inplace_output_transform(field_type *field);

UTIL_IS_INSTANCE_HEADER(field);
UTIL_SAFE_CAST_HEADER_CONST(field);
VOID_ALLOC_HEADER(field);
VOID_FREE_HEADER(field);
VOID_COPY_HEADER(field);
VOID_INITIALIZE_HEADER(field);
VOID_ECL_WRITE_HEADER(field);
VOID_USER_GET_HEADER(field);
VOID_READ_FROM_BUFFER_HEADER(field);
VOID_WRITE_TO_BUFFER_HEADER(field);
VOID_SERIALIZE_HEADER(field);
VOID_DESERIALIZE_HEADER(field);
VOID_FLOAD_HEADER(field);

#endif
