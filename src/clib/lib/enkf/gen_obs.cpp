/*
   See the overview documentation of the observation system in
   enkf_obs.c
*/
#include <cstdlib>
#include <fstream>
#include <vector>

#include <ert/enkf/enkf_macros.hpp>
#include <ert/enkf/gen_obs.hpp>
#include <ert/except.hpp>
#include <ert/python.hpp>
#include <ert/util/string_util.h>
#include <ert/util/util.h>
#include <filesystem>

/**
   This file implemenets a structure for general observations. A
   general observation is just a vector of numbers - where EnKF has no
   understanding whatsover of the type of these data. The actual data
   is supposed to be found in a file.

   Currently it can only observe gen_data instances - but that should
   be generalized.

  The std_scaling field of the xxx_obs structure can be used to scale
  the standard deviation used for the observations, either to support
  workflows with multiple data assimilation or to reduce the effect of
  observation correlations.

  When querying for the observation standard deviation using
  gen_obs_iget_std() the user input value of standard deviation will
  be returned.
*/
struct gen_obs_struct {
    /** This is the total size of the observation vector. */
    int obs_size;
    /** The indexes which are observed in the corresponding gen_data instance -
     * of length obs_size. */
    std::vector<int> data_index_list;
    /** Flag which indiactes whether all data in the gen_data instance should
     * be observed - in that case we must do a size comparizon-check at use time.*/
    bool observe_all_data;

    /** The observed data. */
    double *obs_data;
    /** The observed standard deviation. */
    double *obs_std;
    /** Scaling factor for the standard deviation */
    double *std_scaling;

    /** The key this observation is held by - in the enkf_obs structur (only
     * for debug messages). */
    char *obs_key;
};

void gen_obs_free(gen_obs_type *gen_obs) {
    free(gen_obs->obs_data);
    free(gen_obs->obs_std);
    free(gen_obs->obs_key);
    free(gen_obs->std_scaling);

    delete gen_obs;
}

/**
   This function loads the actual observations from disk, and
   initializes the obs_data and obs_std pointers with the
   observations. It also sets the obs_size field of the gen_obs
   instance.

   The file with observations should be a long vector of 2N elements,
   where the first N elements are data values, and the last N values
   are the corresponding standard deviations.
*/
static void gen_obs_set_data(gen_obs_type *gen_obs, int buffer_size,
                             const double *buffer) {
    gen_obs->obs_size = buffer_size / 2;
    gen_obs->obs_data = (double *)util_realloc(
        gen_obs->obs_data, gen_obs->obs_size * sizeof *gen_obs->obs_data);
    gen_obs->obs_std = (double *)util_realloc(
        gen_obs->obs_std, gen_obs->obs_size * sizeof *gen_obs->obs_std);
    gen_obs->std_scaling = (double *)util_realloc(
        gen_obs->std_scaling, gen_obs->obs_size * sizeof *gen_obs->std_scaling);
    gen_obs->data_index_list.resize(gen_obs->obs_size);

    for (int iobs = 0; iobs < gen_obs->obs_size; iobs++) {
        gen_obs->obs_data[iobs] = buffer[2 * iobs];
        gen_obs->obs_std[iobs] = buffer[2 * iobs + 1];
        gen_obs->std_scaling[iobs] = 1.0;
        gen_obs->data_index_list[iobs] = iobs;
    }
}

void gen_obs_load_observation(gen_obs_type *gen_obs, const char *obs_file) {
    const std::filesystem::path &path = obs_file;
    std::ifstream stream{path};
    stream.imbue(std::locale::classic());

    std::vector<double> data;
    for (;;) {
        double value;
        if (!(stream >> value))
            break;
        data.emplace_back(value);
        stream >> std::ws;
    }
    if (!stream.eof())
        throw exc::runtime_error{
                "Could not parse contents of {} as a sequence of numbers", path};

    auto vec = data;
    gen_obs_set_data(gen_obs, vec.size(), vec.data());
}

void gen_obs_set_scalar(gen_obs_type *gen_obs, double scalar_value,
                        double scalar_std) {
    double buffer[2] = {scalar_value, scalar_std};
    gen_obs_set_data(gen_obs, 2, buffer);
}

void gen_obs_attach_data_index(gen_obs_type *obs,
                               const int_vector_type *data_index) {
    size_t size = int_vector_size(data_index);
    obs->data_index_list.resize(size);
    for (size_t i{}; i < size; ++i)
        obs->data_index_list[i] = int_vector_iget(data_index, i);
    obs->observe_all_data = false;
}

void gen_obs_load_data_index(gen_obs_type *obs, const char *data_index_file) {
    std::ifstream stream{data_index_file};
    stream.imbue(std::locale::classic());

    obs->data_index_list.clear();
    while (stream) {
        int value;
        if (!(stream >> value))
            break;
        obs->data_index_list.emplace_back(value);
        stream >> std::ws;
    }
    if (!stream.eof())
        throw exc::runtime_error{
            "Failure during parsing of gen_obs data index file {}",
            data_index_file};

    obs->obs_size = obs->data_index_list.size();
    obs->observe_all_data = false;
}

gen_obs_type *gen_obs_alloc__(const char *obs_key) {
    auto obs = new gen_obs_type;
    obs->obs_data = NULL;
    obs->obs_std = NULL;
    obs->std_scaling = NULL;
    obs->obs_key = util_alloc_string_copy(obs_key);
    obs->observe_all_data = true;
    return obs;
}

/*
   In general the gen_obs observation vector can be smaller than the
   gen_data field it is observing, i.e. we can have a situation like
   this:

           Data               Obs
           ----               ---

          [ 6.0 ] ----\
          [ 2.0 ]      \---> [ 6.3 ]
          [ 3.0 ] ---------> [ 2.8 ]
          [ 2.0 ]      /---> [ 4.3 ]
          [ 4.5 ] ----/

   The situation here is as follows:

   1. We have a gen data vector with five elements.

   2. We have an observation vector of three elements, which observes
      three of the elements in the gen_data vector, in this particular
      case the data_index_list of the observation equals: [0 , 2 , 4].

   Now when we want to look at the match of observation quality of the
   last element in the observation vector it would be natural to use
   the user_get key: "obs_key:2" - however this is an observation of
   data element number 4, i.e. as seen from data context (when adding
   observations to an ensemble plot) the natural indexing would be:
   "data_key:4".


   The function gen_obs_user_get_with_data_index() will do the
   translation from data based indexing to observation based indexing, i.e.

      gen_obs_user_get_with_data_index("4")

   will do an inverse lookup of the '4' and further call

      gen_obs_user_get("2")

*/

void gen_obs_user_get(const gen_obs_type *gen_obs, const char *index_key,
                      double *value, double *std, bool *valid) {
    int index;
    *valid = false;

    if (util_sscanf_int(index_key, &index)) {
        if ((index >= 0) && (index < gen_obs->obs_size)) {
            *valid = true;
            *value = gen_obs->obs_data[index];
            *std = gen_obs->obs_std[index];
        }
    }
}

void gen_obs_user_get_with_data_index(const gen_obs_type *gen_obs,
                                      const char *index_key, double *value,
                                      double *std, bool *valid) {
    if (gen_obs->observe_all_data)
        /* The observation and data vectors are equally long - no reverse lookup necessary. */
        gen_obs_user_get(gen_obs, index_key, value, std, valid);
    else {
        *valid = false;
        int data_index;
        if (util_sscanf_int(index_key, &data_index)) {
            int obs_index = 0;
            do {
                if (gen_obs->data_index_list[obs_index] == data_index)
                    /* Found it - will use the 'obs_index' value. */
                    break;

                obs_index++;
            } while (obs_index < gen_obs->obs_size);
            if (obs_index <
                gen_obs->obs_size) { /* The reverse lookup succeeded. */
                *valid = true;
                *value = gen_obs->obs_data[obs_index];
                *std = gen_obs->obs_std[obs_index];
            }
        }
    }
}

void gen_obs_update_std_scale(gen_obs_type *gen_obs, double std_multiplier,
                              const ActiveList *active_list) {
    if (active_list->getMode() == ALL_ACTIVE) {
        for (int i = 0; i < gen_obs->obs_size; i++)
            gen_obs->std_scaling[i] = std_multiplier;
    } else {
        const int *active_index = active_list->active_list_get_active();
        int size = active_list->active_size(gen_obs->obs_size);
        for (int i = 0; i < size; i++) {
            int obs_index = active_index[i];
            if (obs_index >= gen_obs->obs_size) {
                util_abort("[Gen_Obs] Index out of bounds %d [0, %d]",
                           obs_index, gen_obs->obs_size - 1);
            }
            gen_obs->std_scaling[obs_index] = std_multiplier;
        }
    }
}

int gen_obs_get_size(const gen_obs_type *gen_obs) { return gen_obs->obs_size; }

double gen_obs_iget_std(const gen_obs_type *gen_obs, int index) {
    return gen_obs->obs_std[index];
}

double gen_obs_iget_std_scaling(const gen_obs_type *gen_obs, int index) {
    return gen_obs->std_scaling[index];
}

double gen_obs_iget_value(const gen_obs_type *gen_obs, int index) {
    return gen_obs->obs_data[index];
}

void gen_obs_load_values(const gen_obs_type *gen_obs, int size, double *data) {
    for (int i = 0; i < size; i++) {
        data[i] = gen_obs->obs_data[i];
    }
}

void gen_obs_load_std(const gen_obs_type *gen_obs, int size, double *data) {
    for (int i = 0; i < size; i++) {
        data[i] = gen_obs->obs_std[i];
    }
}

int gen_obs_get_obs_index(const gen_obs_type *gen_obs, int index) {
    if (index < 0 || index >= gen_obs->obs_size) {
        util_abort("[Gen_Obs] Index out of bounds %d [0, %d]", index,
                   gen_obs->obs_size - 1);
    }

    if (gen_obs->observe_all_data) {
        return index;
    } else {
        return gen_obs->data_index_list[index];
    }
}

VOID_FREE(gen_obs)
VOID_USER_GET_OBS(gen_obs)
VOID_UPDATE_STD_SCALE(gen_obs)

class ActiveList;
namespace {
void update_std_scaling(Cwrap<gen_obs_type> self, double scaling,
                        const ActiveList &active_list) {
    gen_obs_update_std_scale(self, scaling, &active_list);
}
} // namespace

ERT_CLIB_SUBMODULE("local.gen_obs", m) {
    using namespace py::literals;

    m.def("update_std_scaling", &update_std_scaling, "self"_a, "scaling"_a,
          "active_list"_a);
}
