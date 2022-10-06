/*
   See the overview documentation of the observation system in enkf_obs.c
*/
#include <stdlib.h>

#include <ert/util/util.h>

#include <ert/enkf/summary_obs.hpp>

#include "ert/python.hpp"

#define SUMMARY_OBS_TYPE_ID 66103
#define OBS_SIZE 1

struct summary_obs_struct {
    UTIL_TYPE_ID_DECLARATION;
    /** The observation, in summary.x syntax, e.g. GOPR:FIELD. */
    char *summary_key;
    char *obs_key;

    /** Observation value. */
    double value;
    /** Standard deviation of observation. */
    double std;
    double std_scaling;
};

/**
  This function allocates a summary_obs instance. The summary_key
  string should be of the format used by the summary.x program.
  E.g., WOPR:P4 would condition on WOPR in well P4.

  Observe that this format is currently *not* checked before the actual
  observation time.

  TODO
  Should check summary_key on alloc.
*/
summary_obs_type *summary_obs_alloc(const char *summary_key,
                                    const char *obs_key, double value,
                                    double std) {

    summary_obs_type *obs = (summary_obs_type *)util_malloc(sizeof *obs);
    UTIL_TYPE_ID_INIT(obs, SUMMARY_OBS_TYPE_ID)

    obs->summary_key = util_alloc_string_copy(summary_key);
    obs->obs_key = util_alloc_string_copy(obs_key);
    obs->value = value;
    obs->std = std;
    obs->std_scaling = 1.0;

    return obs;
}

static UTIL_SAFE_CAST_FUNCTION_CONST(summary_obs, SUMMARY_OBS_TYPE_ID);
static UTIL_SAFE_CAST_FUNCTION(summary_obs, SUMMARY_OBS_TYPE_ID);
UTIL_IS_INSTANCE_FUNCTION(summary_obs, SUMMARY_OBS_TYPE_ID);

void summary_obs_free(summary_obs_type *summary_obs) {
    free(summary_obs->summary_key);
    free(summary_obs->obs_key);
    free(summary_obs);
}

const char *summary_obs_get_summary_key(const summary_obs_type *summary_obs) {
    return summary_obs->summary_key;
}

void summary_obs_get_observations(const summary_obs_type *summary_obs,
                                  obs_data_type *obs_data, enkf_fs_type *fs,
                                  int report_step) {

    obs_block_type *obs_block =
        obs_data_add_block(obs_data, summary_obs->obs_key, OBS_SIZE);
    obs_block_iset(obs_block, 0, summary_obs->value,
                   summary_obs->std * summary_obs->std_scaling);
}

void summary_obs_measure(const summary_obs_type *obs,
                         const summary_type *summary, node_id_type node_id,
                         meas_data_type *meas_data) {
    {
        meas_block_type *meas_block = meas_data_add_block(
            meas_data, obs->obs_key, node_id.report_step, OBS_SIZE);
        meas_block_iset(meas_block, node_id.iens, 0,
                        summary_get(summary, node_id.report_step));
    }
}

C_USED double summary_obs_chi2(const summary_obs_type *obs,
                               const summary_type *summary,
                               node_id_type node_id) {
    double x =
        (summary_get(summary, node_id.report_step) - obs->value) / obs->std;
    return x * x;
}

void summary_obs_user_get(const summary_obs_type *summary_obs,
                          const char *index_key, double *value, double *std,
                          bool *valid) {
    *valid = true;
    *value = summary_obs->value;
    *std = summary_obs->std;
}

double summary_obs_get_value(const summary_obs_type *summary_obs) {
    return summary_obs->value;
}

double summary_obs_get_std(const summary_obs_type *summary_obs) {
    return summary_obs->std;
}

double summary_obs_get_std_scaling(const summary_obs_type *summary_obs) {
    return summary_obs->std_scaling;
}

void summary_obs_update_std_scale(summary_obs_type *summary_obs,
                                  double std_multiplier,
                                  const ActiveList *active_list) {
    if (active_list->getMode() == ALL_ACTIVE)
        summary_obs->std_scaling = std_multiplier;
    else {
        int size = active_list->active_size(OBS_SIZE);
        if (size > 0)
            summary_obs->std_scaling = std_multiplier;
    }
}

void summary_obs_set_std_scale(summary_obs_type *summary_obs,
                               double std_multiplier) {
    summary_obs->std_scaling = std_multiplier;
}

VOID_FREE(summary_obs)
VOID_GET_OBS(summary_obs)
VOID_USER_GET_OBS(summary_obs)
VOID_MEASURE(summary_obs, summary)
VOID_CHI2(summary_obs, summary)
VOID_UPDATE_STD_SCALE(summary_obs);

class ActiveList;
namespace {
void update_std_scaling(py::handle obj, double scaling,
                        const ActiveList &active_list) {
    auto *summary_obs = reinterpret_cast<summary_obs_type *>(
        PyLong_AsVoidPtr(obj.attr("_BaseCClass__c_pointer").ptr()));
    summary_obs_update_std_scale(summary_obs, scaling, &active_list);
}
} // namespace

ERT_CLIB_SUBMODULE("local.summary_obs", m) {
    using namespace py::literals;

    m.def("update_std_scaling", &update_std_scaling, "self"_a, "scaling"_a,
          "active_list"_a);
}
