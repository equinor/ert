/*
   See the overview documentation of the observation system in enkf_obs.c
*/

#include <algorithm>
#include <cmath>
#include <cppitertools/enumerate.hpp>
#include <ert/except.hpp>
#include <ert/logging.hpp>
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

static int find_nearest_time_index(std::vector<time_t> time_map,
                                   time_t external_time) {
    int nearest_index = -1;
    auto nearest_diff = std::numeric_limits<double>::infinity();
    for (auto &&[index, time] : iter::enumerate(time_map)) {
        time_t diff = std::abs(external_time - time);
        if (diff == 0)
            return index;
        /** The tolerance of 30s is arbitrarily chosen */
        if (diff < 30 && diff < nearest_diff) {
            nearest_diff = diff;
            nearest_index = index;
        }
    }
    return nearest_index;
}

static int __conf_instance_get_restart_nr(
    std::shared_ptr<conf_instance_type> conf_instance, const char *obs_key,
    const std::vector<time_t> &time_map) {
    int obs_restart_nr = -1;

    if (conf_instance_has_item(conf_instance, "RESTART")) {
        obs_restart_nr =
            conf_instance_get_item_value_int(conf_instance, "RESTART");
        if (obs_restart_nr >= time_map.size())
            throw exc::out_of_range(
                "Observation: {} at restart {}, but history "
                "file has only {} restarts.",
                obs_key, obs_restart_nr, time_map.size());
        return obs_restart_nr;
    }

    time_t obs_time = time_map.at(0);
    if (conf_instance_has_item(conf_instance, "DATE")) {
        obs_time = conf_instance_get_item_value_time_t(conf_instance, "DATE");
    } else if (conf_instance_has_item(conf_instance, "DAYS")) {
        double days =
            conf_instance_get_item_value_double(conf_instance, "DAYS");
        util_inplace_forward_days_utc(&obs_time, days);
    } else if (conf_instance_has_item(conf_instance, "HOURS")) {
        double hours =
            conf_instance_get_item_value_double(conf_instance, "HOURS");
        util_inplace_forward_seconds_utc(&obs_time, hours * 3600);
    } else
        throw exc::runtime_error("Observation: {} missing DATE, DAYS or HOURS",
                                 obs_key);
    obs_restart_nr = find_nearest_time_index(time_map, obs_time);
    if (obs_restart_nr < 0) {
        std::string error = fmt::format(
            "Observation {} does not have a matching time in the time map.",
            obs_key);
        if (conf_instance_has_item(conf_instance, "DATE"))
            error += fmt::format(
                " DATE={} is not in the time map.",
                conf_instance_get_item_value_ref(conf_instance, "DATE"));
        else if (conf_instance_has_item(conf_instance, "DAYS"))
            error += fmt::format(
                " DAYS={} is not in the time map.",
                conf_instance_get_item_value_ref(conf_instance, "DAYS"));
        throw exc::out_of_range(error);
    }
    return obs_restart_nr;
}

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

void obs_vector_free(obs_vector_type *obs_vector) {
    vector_free(obs_vector->nodes);
    free(obs_vector->obs_key);
    delete obs_vector;
}

void obs_vector_install_node(obs_vector_type *obs_vector, int index,
                             void *node) {
    {
        if (vector_iget_const(obs_vector->nodes, index) == NULL) {
            obs_vector->num_active++;
            obs_vector->step_list.push_back(index);
            std::sort(obs_vector->step_list.begin(),
                      obs_vector->step_list.end());
        }

        vector_iset_owned_ref(obs_vector->nodes, index, node,
                              obs_vector->freef);
    }
}

/**
   Observe that @summary_key is the key used to look up the
   corresponding simulated value in the ensemble, and not the
   observation key - the two can be different.
*/
static void obs_vector_add_summary_obs(obs_vector_type *obs_vector,
                                       int obs_index, const char *summary_key,
                                       const char *obs_key, double value,
                                       double std) {
    summary_obs_type *summary_obs =
        summary_obs_alloc(summary_key, obs_key, value, std);
    obs_vector_install_node(obs_vector, obs_index, summary_obs);
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

/**
   All the obs_vector_load_from_XXXX() functions can safely return
   NULL, in which case no observation is added to enkf_obs observation
   hash table.
*/
void obs_vector_load_from_SUMMARY_OBSERVATION(
    obs_vector_type *obs_vector,
    std::shared_ptr<conf_instance_type> conf_instance,
    const std::vector<time_t> &obs_time) {
    if (!conf_instance_is_of_class(conf_instance, "SUMMARY_OBSERVATION"))
        util_abort("%s: internal error. expected \"SUMMARY_OBSERVATION\" "
                   "instance, got \"%s\".\n",
                   __func__, conf_instance_get_class_name_ref(conf_instance));

    {
        double obs_value =
            conf_instance_get_item_value_double(conf_instance, "VALUE");
        double obs_error =
            conf_instance_get_item_value_double(conf_instance, "ERROR");
        double min_error =
            conf_instance_get_item_value_double(conf_instance, "ERROR_MIN");
        const char *error_mode =
            conf_instance_get_item_value_ref(conf_instance, "ERROR_MODE");
        const char *sum_key =
            conf_instance_get_item_value_ref(conf_instance, "KEY");
        const char *obs_key = conf_instance_get_name_ref(conf_instance);
        int obs_restart_nr =
            __conf_instance_get_restart_nr(conf_instance, obs_key, obs_time);

        if (obs_restart_nr == 0) {
            int day, month, year;
            time_t start_time = obs_time[0];
            util_set_date_values_utc(start_time, &day, &month, &year);

            logger->error("It is unfortunately not possible to use "
                          "summary observations from the start of the "
                          "simulation. Problem with observation '{}' at "
                          "{:4d}-{:02d}-{:02d}",
                          obs_key, year, month, day);
            exit(1);
        }
        {
            if (strcmp(error_mode, "REL") == 0)
                obs_error *= obs_value;
            else if (strcmp(error_mode, "RELMIN") == 0)
                obs_error = util_double_max(min_error, obs_error * obs_value);

            obs_vector_add_summary_obs(obs_vector, obs_restart_nr, sum_key,
                                       obs_key, obs_value, obs_error);
        }
    }
}

obs_vector_type *obs_vector_alloc_from_GENERAL_OBSERVATION(
    std::shared_ptr<conf_instance_type> conf_instance,
    const std::vector<time_t> &obs_time, enkf_config_node_type *config_node) {
    if (!conf_instance_is_of_class(conf_instance, "GENERAL_OBSERVATION"))
        util_abort("%s: internal error. expected \"GENERAL_OBSERVATION\" "
                   "instance, got \"%s\".\n",
                   __func__, conf_instance_get_class_name_ref(conf_instance));
    const char *obs_key = conf_instance_get_name_ref(conf_instance);
    const char *state_kw =
        conf_instance_get_item_value_ref(conf_instance, "DATA");

    int obs_restart_nr =
        __conf_instance_get_restart_nr(conf_instance, obs_key, obs_time);
    const char *index_file = NULL;
    const char *index_list = NULL;
    const char *obs_file = NULL;

    if (conf_instance_has_item(conf_instance, "INDEX_FILE"))
        index_file =
            conf_instance_get_item_value_ref(conf_instance, "INDEX_FILE");

    if (conf_instance_has_item(conf_instance, "INDEX_LIST"))
        index_list =
            conf_instance_get_item_value_ref(conf_instance, "INDEX_LIST");

    if (conf_instance_has_item(conf_instance, "OBS_FILE"))
        obs_file = conf_instance_get_item_value_ref(conf_instance, "OBS_FILE");

    {
        obs_vector_type *obs_vector = NULL;

        if (enkf_config_node_get_impl_type(config_node) == GEN_DATA) {
            double scalar_error = -1;
            double scalar_value = -1;
            gen_obs_type *gen_obs;
            const gen_data_config_type *config =
                (const gen_data_config_type *)enkf_config_node_get_ref(
                    config_node);

            if (gen_data_config_has_report_step(config, obs_restart_nr)) {
                obs_vector = obs_vector_alloc(GEN_OBS, obs_key, config_node,
                                              obs_time.size());
                if (conf_instance_has_item(conf_instance, "VALUE")) {
                    scalar_value = conf_instance_get_item_value_double(
                        conf_instance, "VALUE");
                    scalar_error = conf_instance_get_item_value_double(
                        conf_instance, "ERROR");
                }

                /* The config system has ensured that we have either OBS_FILE or (VALUE and ERROR). */
                gen_obs = gen_obs_alloc(
                    (const gen_data_config_type *)enkf_config_node_get_ref(
                        config_node),
                    obs_key, obs_file, scalar_value, scalar_error, index_file,
                    index_list);
                obs_vector_install_node(obs_vector, obs_restart_nr, gen_obs);
            } else
                fprintf(stderr,
                        "** ERROR: The GEN_DATA node:%s is not configured "
                        "to load from report step:%d - the observation:%s "
                        "will be ignored\n",
                        state_kw, obs_restart_nr, obs_key);
        } else {
            ert_impl_type impl_type =
                enkf_config_node_get_impl_type(config_node);
            fprintf(stderr,
                    "** ERROR: %s: %s has implementation type:\'%s\' - "
                    "expected:\'%s\' - observation:%s ignored.\n",
                    __func__, state_kw, enkf_types_get_impl_name(impl_type),
                    enkf_types_get_impl_name(GEN_DATA), obs_key);
        }
        return obs_vector;
    }
}

/**
 * @brief Reads history from ecl_summary file for use as ert observation
 *
 *  Depending on the value of history_source_type the key that is being
 *  read is either the result key (for example FOPR), or the historical
 *  key (in this example FOPRH), both of which can be stored in an ecl
 *  summary file
 *
 * @param history Which type of key to read
 * @param value Values are loaded into
 * @param summary_key Base name of the key to read
 * @param valid Checks if values are valid
 * @param refcase ecl summary file
 */
static bool read_history_from_ecl_summary(const history_source_type history,
                                          const char *summary_key,
                                          double_vector_type *value,
                                          bool_vector_type *valid,
                                          const ecl_sum_type *refcase) {
    bool initOK = false;

    double_vector_reset(value);
    bool_vector_reset(valid);
    bool_vector_set_default(valid, false);

    std::optional<std::string> local_key = std::nullopt;
    if (history == REFCASE_HISTORY) {
        /* Must create a new key with 'H' for historical values. */
        const ecl_smspec_type *smspec = ecl_sum_get_smspec(refcase);
        const char *join_string = ecl_smspec_get_join_string(smspec);
        ecl_smspec_var_type var_type =
            ecl_smspec_identify_var_type(summary_key);

        if ((var_type == ECL_SMSPEC_WELL_VAR) ||
            (var_type == ECL_SMSPEC_GROUP_VAR))
            local_key = std::string(std::move(util_alloc_sprintf(
                "%sH%s%s", ecl_sum_get_keyword(refcase, summary_key),
                join_string, ecl_sum_get_wgname(refcase, summary_key))));
        else if (var_type == ECL_SMSPEC_FIELD_VAR)
            local_key = std::string(std::move(util_alloc_sprintf(
                "%sH", ecl_sum_get_keyword(refcase, summary_key))));
    } else
        local_key = std::string(summary_key);

    if (local_key.has_value()) {
        if (ecl_sum_has_general_var(refcase, local_key.value().c_str())) {
            for (int tstep = 0; tstep <= ecl_sum_get_last_report_step(refcase);
                 tstep++) {
                if (ecl_sum_has_report_step(refcase, tstep)) {
                    int time_index = ecl_sum_iget_report_end(refcase, tstep);
                    double_vector_iset(
                        value, tstep,
                        ecl_sum_get_general_var(refcase, time_index,
                                                local_key.value().c_str()));
                    bool_vector_iset(valid, tstep, true);
                } else
                    bool_vector_iset(valid, tstep,
                                     false); /* Did not have this report step */
            }
            initOK = true;
        }
    }
    return initOK;
}

// Should check the refcase for key - if it is != NULL.

bool obs_vector_load_from_HISTORY_OBSERVATION(
    obs_vector_type *obs_vector,
    std::shared_ptr<conf_instance_type> conf_instance,
    const std::vector<time_t> &obs_time, const history_source_type history,
    double std_cutoff, const ecl_sum_type *refcase) {

    if (!conf_instance_is_of_class(conf_instance, "HISTORY_OBSERVATION"))
        util_abort("%s: internal error. expected \"HISTORY_OBSERVATION\" "
                   "instance, got \"%s\".\n",
                   __func__, conf_instance_get_class_name_ref(conf_instance));

    {
        bool initOK = false;
        double_vector_type *value = double_vector_alloc(0, 0);
        double_vector_type *std = double_vector_alloc(0, 0);
        bool_vector_type *valid = bool_vector_alloc(0, false);

        double error =
            conf_instance_get_item_value_double(conf_instance, "ERROR");
        double error_min =
            conf_instance_get_item_value_double(conf_instance, "ERROR_MIN");
        const char *error_mode =
            conf_instance_get_item_value_ref(conf_instance, "ERROR_MODE");
        const char *sum_key = conf_instance_get_name_ref(conf_instance);

        // Get time series data from refcase and allocate
        if (read_history_from_ecl_summary(history, sum_key, value, valid,
                                          refcase)) {
            // Create  the standard deviation vector
            if (strcmp(error_mode, "ABS") == 0) {
                for (size_t restart_nr = 0; restart_nr < obs_time.size();
                     restart_nr++)
                    double_vector_iset(std, restart_nr, error);
            } else if (strcmp(error_mode, "REL") == 0) {
                for (size_t restart_nr = 0; restart_nr < obs_time.size();
                     restart_nr++)
                    double_vector_iset(std, restart_nr,
                                       error * std::abs(double_vector_iget(
                                                   value, restart_nr)));
            } else if (strcmp(error_mode, "RELMIN") == 0) {
                for (size_t restart_nr = 0; restart_nr < obs_time.size();
                     restart_nr++) {
                    double tmp_std = util_double_max(
                        error_min, error * std::abs(double_vector_iget(
                                               value, restart_nr)));
                    double_vector_iset(std, restart_nr, tmp_std);
                }
            } else
                util_abort("%s: Internal error. Unknown error mode \"%s\"\n",
                           __func__, error_mode);

            // Handle SEGMENTs which can be used to customize the observation error. */
            {
                std::vector<std::string> segment_keys =
                    conf_instance_alloc_list_of_sub_instances_of_class_by_name(
                        conf_instance, "SEGMENT");
                std::sort(segment_keys.begin(), segment_keys.end());

                int num_segments = segment_keys.size();

                for (int segment_nr = 0; segment_nr < num_segments;
                     segment_nr++) {
                    std::string segment_name = segment_keys[segment_nr];
                    auto segment_conf = conf_instance_get_sub_instance_ref(
                        conf_instance, segment_name.c_str());

                    int start =
                        conf_instance_get_item_value_int(segment_conf, "START");
                    int stop =
                        conf_instance_get_item_value_int(segment_conf, "STOP");
                    double error_segment = conf_instance_get_item_value_double(
                        segment_conf, "ERROR");
                    double error_min_segment =
                        conf_instance_get_item_value_double(segment_conf,
                                                            "ERROR_MIN");
                    const char *error_mode_segment =
                        conf_instance_get_item_value_ref(segment_conf,
                                                         "ERROR_MODE");

                    if (start < 0) {
                        printf("%s: WARNING - Segment out of bounds. "
                               "Truncating start of segment to 0.\n",
                               __func__);
                        start = 0;
                    }

                    if (stop >= obs_time.size()) {
                        fmt::print("%s: WARNING - Segment out of bounds. "
                                   "Truncating end of segment to %ld.\n",
                                   __func__, obs_time.size() - 1);
                        stop = obs_time.size() - 1;
                    }

                    if (start > stop) {
                        printf("%s: WARNING - Segment start after stop. "
                               "Truncating end of segment to %d.\n",
                               __func__, start);
                        stop = start;
                    }

                    // Create  the standard deviation vector
                    if (strcmp(error_mode_segment, "ABS") == 0) {
                        for (size_t restart_nr = start; restart_nr <= stop;
                             restart_nr++)
                            double_vector_iset(std, restart_nr, error_segment);
                    } else if (strcmp(error_mode_segment, "REL") == 0) {
                        for (size_t restart_nr = start; restart_nr <= stop;
                             restart_nr++)
                            double_vector_iset(std, restart_nr,
                                               error_segment *
                                                   std::abs(double_vector_iget(
                                                       value, restart_nr)));
                    } else if (strcmp(error_mode_segment, "RELMIN") == 0) {
                        for (size_t restart_nr = start; restart_nr <= stop;
                             restart_nr++) {
                            double tmp_std = util_double_max(
                                error_min_segment,
                                error_segment * std::abs(double_vector_iget(
                                                    value, restart_nr)));
                            double_vector_iset(std, restart_nr, tmp_std);
                        }
                    } else
                        util_abort(
                            "%s: Internal error. Unknown error mode \"%s\"\n",
                            __func__, error_mode);
                }
            }

            /*
        This is where the summary observations are finally added.
      */
            for (size_t restart_nr = 0; restart_nr < obs_time.size();
                 restart_nr++) {
                if (bool_vector_safe_iget(valid, restart_nr)) {
                    if (double_vector_iget(std, restart_nr) > std_cutoff) {
                        obs_vector_add_summary_obs(
                            obs_vector, restart_nr, sum_key, sum_key,
                            double_vector_iget(value, restart_nr),
                            double_vector_iget(std, restart_nr));
                    } else
                        logger->warning("Too small observation error in "
                                        "observation {}:{} - ignored",
                                        sum_key, restart_nr);
                }
            }
            initOK = true;
        }
        double_vector_free(std);
        double_vector_free(value);
        bool_vector_free(valid);
        return initOK;
    }
}

static const char *__summary_kw(const char *field_name) {
    if (strcmp(field_name, "PRESSURE") == 0)
        return "BPR";
    else if (strcmp(field_name, "SWAT") == 0)
        return "BSWAT";
    else if (strcmp(field_name, "SGAS") == 0)
        return "BSGAS";
    else {
        util_abort("%s: sorry - could not \'translate\' field:%s to block "
                   "summary variable\n",
                   __func__, field_name);
        return NULL;
    }
}

const char *obs_vector_get_obs_key(const obs_vector_type *obs_vector) {
    return obs_vector->obs_key;
}

VOID_FREE(obs_vector)
