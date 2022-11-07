#ifndef ERT_SUMMARY_H
#define ERT_SUMMARY_H
#include <ert/util/double_vector.h>

#include <ert/ecl/ecl_file.h>
#include <ert/ecl/ecl_sum.h>

#include <ert/enkf/enkf_macros.hpp>
#include <ert/enkf/enkf_util.hpp>
#include <ert/enkf/summary_config.hpp>

extern "C" summary_type *
summary_alloc(const summary_config_type *summary_config);
extern "C" void summary_free(summary_type *summary);
extern "C" double summary_get(const summary_type *summary, int report_step);
extern "C" void summary_set(summary_type *summary, int report_step,
                            double value);
bool summary_active_value(double value);
extern "C" int summary_length(const summary_type *summary);
extern "C" double summary_undefined_value();
std::vector<double> summary_user_get_vector(const summary_type *summary);

VOID_HAS_DATA_HEADER(summary);
VOID_ALLOC_HEADER(summary);
VOID_FREE_HEADER(summary);
VOID_FORWARD_LOAD_HEADER(summary);
VOID_FORWARD_LOAD_VECTOR_HEADER(summary);
VOID_USER_GET_HEADER(summary);
VOID_WRITE_TO_BUFFER_HEADER(summary);
VOID_READ_FROM_BUFFER_HEADER(summary);
VOID_SERIALIZE_HEADER(summary)
VOID_DESERIALIZE_HEADER(summary)

#endif
