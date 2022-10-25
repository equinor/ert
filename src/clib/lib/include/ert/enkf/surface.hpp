#ifndef ERT_SURFACE_H
#define ERT_SURFACE_H

#include <ert/enkf/enkf_macros.hpp>
#include <ert/enkf/enkf_types.hpp>
#include <ert/enkf/surface_config.hpp>

typedef struct surface_struct surface_type;

VOID_ALLOC_HEADER(surface);
VOID_FREE_HEADER(surface);
VOID_USER_GET_HEADER(surface);
VOID_WRITE_TO_BUFFER_HEADER(surface);
VOID_READ_FROM_BUFFER_HEADER(surface);
VOID_SERIALIZE_HEADER(surface);
VOID_DESERIALIZE_HEADER(surface);
VOID_INITIALIZE_HEADER(surface);
VOID_FLOAD_HEADER(surface);

#endif
