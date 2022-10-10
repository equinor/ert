#ifndef ERT_ENKF_UTIL_H
#define ERT_ENKF_UTIL_H
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include <ert/util/buffer.h>

#include <ert/ecl/ecl_type.h>
#include <ert/ecl/ecl_util.h>

#include <ert/enkf/enkf_types.hpp>

inline void enkf_util_assert_buffer_type(buffer_type *buffer,
                                         ert_impl_type target_type) {
    auto type = static_cast<ert_impl_type>(buffer_fread_int(buffer));
    if (type != target_type) {
        util_abort(
            "%s: wrong target type in file (expected:%d  got:%d) - aborting \n",
            __func__, target_type, type);
    }
}

#endif
