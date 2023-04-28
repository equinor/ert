/*
   See the overview documentation of the observation system in enkf_obs.c
*/

#include <algorithm>
#include <cmath>
#include <cppitertools/enumerate.hpp>
#include <ert/except.hpp>
#include <ert/logging.hpp>
#include <ert/python.hpp>
#include <ert/util/bool_vector.h>
#include <ert/util/double_vector.h>
#include <ert/util/util.h>
#include <ert/util/vector.h>
#include <optional>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>

#include <ert/config/conf.hpp>

#include <ert/ecl/ecl_grid.h>

#include <ert/enkf/active_list.hpp>
#include <ert/enkf/enkf_defaults.hpp>
#include <ert/enkf/gen_obs.hpp>
#include <ert/enkf/obs_vector.hpp>
#include <ert/enkf/summary_obs.hpp>

static auto logger = ert::get_logger("obs_vector");

struct obs_vector_struct {
    /** Function used to free an observation node. */
    obs_free_ftype *freef;
    /** Function to get an observation based on KEY:INDEX input from user.*/
    obs_user_get_ftype *user_get;
    /** Function to scale the standard deviation with a given factor */
    obs_update_std_scale_ftype *update_std_scale;
    vector_type *nodes;
    /** The key this observation vector has in the enkf_obs layer. */
    char *obs_key;
    /** The config_node of the node type we are observing - shared reference */
    enkf_config_node_type *config_node;
    obs_impl_type obs_type;
    /** The total number of timesteps where this observation is active (i.e.
     * nodes[ ] != NULL) */
    int num_active;
    std::vector<int> step_list;
};
static void obs_vector_resize(obs_vector_type *vector, int new_size) {
    int current_size = vector_get_size(vector->nodes);
    int i;

    for (i = current_size; i < new_size; i++)
        vector_append_ref(vector->nodes, NULL);
}

obs_vector_type *obs_vector_alloc(obs_impl_type obs_type, const char *obs_key,
                                  enkf_config_node_type *config_node,
                                  size_t num_reports) {
    auto vector = new obs_vector_type;

    vector->freef = NULL;
    vector->user_get = NULL;
    vector->update_std_scale = NULL;

    switch (obs_type) {
    case (SUMMARY_OBS):
        vector->freef = summary_obs_free__;
        vector->user_get = summary_obs_user_get__;
        vector->update_std_scale = summary_obs_update_std_scale__;
        break;
    case (GEN_OBS):
        vector->freef = gen_obs_free__;
        vector->user_get = gen_obs_user_get__;
        vector->update_std_scale = gen_obs_update_std_scale__;
        break;
    default:
        util_abort("%s: internal error - obs_type:%d not recognized \n",
                   __func__, obs_type);
    }

    vector->obs_type = obs_type;
    vector->config_node = config_node;
    vector->obs_key = util_alloc_string_copy(obs_key);
    vector->num_active = 0;
    vector->nodes = vector_alloc_new();
    obs_vector_resize(vector, num_reports);

    return vector;
}

obs_impl_type obs_vector_get_impl_type(const obs_vector_type *obs_vector) {
    return obs_vector->obs_type;
}

/**
   This is the key for the enkf_node which this observation is
   'looking at'. I.e. if this observation is an RFT pressure
   measurement, this function will return "PRESSURE".
*/
const char *obs_vector_get_state_kw(const obs_vector_type *obs_vector) {
    return enkf_config_node_get_key(obs_vector->config_node);
}

const char *obs_vector_get_key(const obs_vector_type *obs_vector) {
    return obs_vector->obs_key;
}

enkf_config_node_type *
obs_vector_get_config_node(const obs_vector_type *obs_vector) {
    return obs_vector->config_node;
}

void obs_vector_free(obs_vector_type *obs_vector) {
    vector_free(obs_vector->nodes);
    free(obs_vector->obs_key);
    delete obs_vector;
}

void obs_vector_install_node(obs_vector_type *obs_vector, int index,
                             void *node) {
    if (vector_iget_const(obs_vector->nodes, index) == NULL) {
        obs_vector->num_active++;
        obs_vector->step_list.push_back(index);
        std::sort(obs_vector->step_list.begin(), obs_vector->step_list.end());
    }

    vector_iset_owned_ref(obs_vector->nodes, index, node, obs_vector->freef);
}

int obs_vector_get_num_active(const obs_vector_type *vector) {
    return vector->num_active;
}

const std::vector<int> &
obs_vector_get_step_list(const obs_vector_type *vector) {
    return vector->step_list;
}

bool obs_vector_iget_active(const obs_vector_type *vector, int index) {
    /* We accept this ... */
    if (index >= vector_get_size(vector->nodes))
        return false;

    {
        void *obs_data = (void *)vector_iget(vector->nodes, index);
        if (obs_data != NULL)
            return true;
        else
            return false;
    }
}

/**
   Will happily return NULL if index is not active.
*/
void *obs_vector_iget_node(const obs_vector_type *vector, int index) {
    return vector_iget(vector->nodes, index); // CXX_CAST_ERROR
}

void obs_vector_user_get(const obs_vector_type *obs_vector,
                         const char *index_key, int report_step, double *value,
                         double *std, bool *valid) {
    void *obs_node = obs_vector_iget_node(obs_vector, report_step);
    obs_vector->user_get(obs_node, index_key, value, std, valid);
}

/**
  This function returns the next active (i.e. node != NULL) report
  step, starting with 'prev_step + 1'. If no more active steps are
  found, it will return -1.
*/
int obs_vector_get_next_active_step(const obs_vector_type *obs_vector,
                                    int prev_step) {
    if (prev_step >= (vector_get_size(obs_vector->nodes) - 1))
        return -1;
    else {
        int size = vector_get_size(obs_vector->nodes);
        int next_step = prev_step + 1;
        while ((next_step < size) &&
               (obs_vector_iget_node(obs_vector, next_step) == NULL))
            next_step++;

        if (next_step == size)
            return -1; /* No more active steps. */
        else
            return next_step;
    }
}

const char *obs_vector_get_obs_key(const obs_vector_type *obs_vector) {
    return obs_vector->obs_key;
}

ERT_CLIB_SUBMODULE("obs_vector", m) {
    using namespace py::literals;
    m.def("add_summary_obs",
          [](Cwrap<obs_vector_type> obs_vector,
             Cwrap<summary_obs_type> summary_obs, int obs_index) {
              obs_vector_install_node(obs_vector, obs_index, summary_obs);
          });
    m.def("add_general_obs", [](Cwrap<obs_vector_type> obs_vector,
                                Cwrap<gen_obs_type> gen_obs, int obs_index) {
        obs_vector_install_node(obs_vector, obs_index, gen_obs);
    });
}
VOID_FREE(obs_vector)
