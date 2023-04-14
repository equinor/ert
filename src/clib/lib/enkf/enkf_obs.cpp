#include <cmath>
#include <cppitertools/enumerate.hpp>

#include <ert/config/conf.hpp>
#include <ert/ecl/ecl_grid.h>
#include <ert/ecl/ecl_sum.h>
#include <ert/util/hash.h>
#include <ert/util/type_vector_functions.h>
#include <ert/util/vector.h>

#include <ert/config/conf.hpp>
#include <ert/ecl/ecl_grid.h>
#include <ert/ecl/ecl_sum.h>
#include <ert/enkf/enkf_obs.hpp>
#include <ert/enkf/obs_vector.hpp>
#include <ert/enkf/summary_obs.hpp>
#include <ert/except.hpp>
#include <ert/python.hpp>
#include <ert/res_util/string.hpp>
#include <ert/util/hash.h>
#include <ert/util/type_vector_functions.h>
#include <ert/util/vector.h>

/**

The observation system
----------------------

The observation system in the EnKF code is a three layer system. At
the the top is the enkf_obs_type. The enkf_main object contains one
enkf_obs instance which has internalized ALL the observation data. In
enkf_obs the the data is internalized in a hash table, where the keys
in the table are the keys used the observation file.

The next level is the obs_vector type which is a vector of length
num_report_steps. Each element in this vector can either point a
spesific observation instance (which actually contains the data), or
be NULL, if the observation is not active at this report step. In
addition the obs_vector contains function pointers to manipulate the
observation data at the lowest level.

At the lowest level we have specific observation instances,
field_obs, summary_obs and gen_obs. These instances contain the actual
data.

To summarize we can say:

  1. enkf_obs has ALL the observation data.

  2. obs_vector has the full time series for one observation key,
     i.e. all the watercuts in well P2.

  3. field_obs/gen_obs/summary_obs instances contain the actual
     observed data for one (logical) observation and one report step.


In the following example we have two observations

 WWCT:OP1 The water cut in well OP1. This is an observation which is
    active for many report steps, at the lowest level it is
    implemented as summary_obs.

 RFT_P2 This is an RFT test for one well. Only active at one report
    step, implemented at the lowest level as a field_obs instance.


 In the example below there are in total five report steps, hence all
 the obs_vector instances have five 'slots'. If there is no active
 observation for a particular report step, the corresponding pointer
 in the obs_vector instance is NULL.



      _____________________           _____________________
     /                       enkf_obs                      \
     |                                                     |
     |                                                     |
     | obs_hash: {"WWCT:OP1" , "RFT_P2"}                   |
     |                |           |                        |
     |                |           |                        |
     \________________|___________|________________________/
                      |           |
                      |           |
                      |           |
                      |           \--------------------------------------------------------------\
                      |                                                                          |
                      |                                                                          |
                     \|/                                                                         |
 |--- obs_vector: WWCT:OP1 -----------------------------------------------------|                |
 | Function pointers:       --------  --------  --------  --------  --------    |                |
 | Pointing to the          |      |  |      |  |      |  |      |  |      |    |                |
 | underlying               | NULL |  |  X   |  |  X   |  | NULL |  |  X   |    |                |
 | implementation in the    |      |  |  |   |  |  |   |  |      |  |  |   |    |                |
 | summary_obs object.      --------  ---|----  ---|----  --------  ---|----    |                |
 |---------------------------------------|---------|-------------------|--------|                |
                                         |         |                   |                         |
                                        \|/        |                   |                         |
                                |-- summary_obs -| |                  \|/                        |
                                | Value: 0.56..  | |           |-- summary_obs -|                |
                                | std  : 0.15..  | |           | Value: 0.70..  |                |
                                |----------------| |           | std  : 0.25..  |                |
                                                   |           |----------------|                |
                                                  \|/                                            |
                                          |-- summary_obs -|                                     |
                                          | Value: 0.62..  |                                     |
                                          | std  : 0.12..  |                                     |
                                          |----------------|                                     |
                                                                                                 |
                                                                                                 |
                                                                                                 |
  The observation WWCT:OP1 is an observation of summary type, and the                            |
  obs_vector contains pointers to summary_obs instances; along with                              |
  function pointers to manipulate the summary_obs instances. The                                 |
  observation is not active for report steps 0 and 3, so for these                               |
  report steps the obse vector has a NULL pointer.                                               |
                                                                                                 |
                                                                                                 |
                                                                                                 |
                                                                                                 |
                                                                                                 |
                                                                                                 |
 |--- obs_vector: RFT_P2 -------------------------------------------------------|                |
 | Function pointers:       --------  --------  --------  --------  --------    |                |
 | Pointing to the          |      |  |      |  |      |  |      |  |      |    |<---------------/
 | underlying               | NULL |  | NULL |  | NULL |  |  X   |  | NULL |    |
 | implementation in the    |      |  |      |  |      |  |  |   |  |      |    |
 | field_obs object.        --------  --------  --------  ---|----  --------    |
 |-----------------------------------------------------------|------------------|
                                                             |
                                                             |
                                                            \|/
                                        |-- field_obs -----------------------------------|
                                        | i = 25 , j = 16, k = 10, value = 278, std = 10 |
                                        | i = 25 , j = 16, k = 11, value = 279, std = 10 |
                                        | i = 25 , j = 16, k = 12, value = 279, std = 10 |
                                        | i = 25 , j = 17, k = 12, value = 281, std = 10 |
                                        | i = 25 , j = 18, k = 12, value = 282, std = 10 |
                                        |------------------------------------------------|


 The observation RFT_P2 is an RFT observation which is only active at
 one report step, i.e. 4/5 pointers in the obs_vector are just NULL
 pointers. The actual observation(s) are stored in a field_obs
 instance.

 */
struct enkf_obs_struct {
    /** A hash of obs_vector_types indexed by user provided keys. */
    vector_type *obs_vector;
    hash_type *obs_hash;
    /** For fast lookup of report_step -> obs_time */
    std::vector<time_t> obs_time;

    std::string error;
    /* Several shared resources - can generally be NULL*/
    history_source_type history;
    const ecl_sum_type *refcase;
    ensemble_config_type *ensemble_config;
};

enkf_obs_type *enkf_obs_alloc(const history_source_type history,
                              std::shared_ptr<TimeMap> external_time_map,
                              const ecl_sum_type *refcase,
                              ensemble_config_type *ensemble_config) {
    auto enkf_obs = new enkf_obs_type;
    enkf_obs->obs_hash = hash_alloc();
    enkf_obs->obs_vector = vector_alloc_new();

    enkf_obs->history = history;
    enkf_obs->refcase = refcase;
    enkf_obs->ensemble_config = ensemble_config;
    enkf_obs->error = "";

    /* Initialize obs time: */
    if (enkf_obs->refcase) {
        enkf_obs->obs_time.push_back(ecl_sum_get_start_time(refcase));

        int last_report = ecl_sum_get_last_report_step(refcase);
        for (int step = 1; step <= last_report; step++) {
            auto obs_time = ecl_sum_get_report_time(refcase, step);
            enkf_obs->obs_time.push_back(obs_time);
        }
    } else if (external_time_map) {
        enkf_obs->obs_time = *external_time_map;
    } else {
        enkf_obs->error = "Missing REFCASE or TIMEMAP";
    }

    return enkf_obs;
}

const char *enkf_obs_get_error(const enkf_obs_type *obs) {
    return obs->error.c_str();
}

void enkf_obs_free(enkf_obs_type *enkf_obs) {
    hash_free(enkf_obs->obs_hash);
    vector_free(enkf_obs->obs_vector);
    delete enkf_obs;
}

time_t enkf_obs_iget_obs_time(const enkf_obs_type *enkf_obs, int report_step) {
    if (report_step < 0 || report_step >= enkf_obs->obs_time.size())
        return -1;
    return enkf_obs->obs_time[report_step];
}

/**
   Observe that the obs_vector can be NULL - in which it is of course not added.
*/
void enkf_obs_add_obs_vector(enkf_obs_type *enkf_obs,
                             const obs_vector_type *vector) {

    if (vector != NULL) {
        const char *obs_key = obs_vector_get_key(vector);
        if (hash_has_key(enkf_obs->obs_hash, obs_key))
            throw exc::runtime_error("Duplicate observation: {}", obs_key);

        hash_insert_ref(enkf_obs->obs_hash, obs_key, vector);
        vector_append_owned_ref(enkf_obs->obs_vector, vector,
                                obs_vector_free__);
    }
}

bool enkf_obs_has_key(const enkf_obs_type *obs, const char *key) {
    return hash_has_key(obs->obs_hash, key);
}

/** @brief get the observation vector for the given observation key.
 *
 * @param obs The enkf_obs_type object.
 * @param key The observation key to get observation vector for.
 * @return The observation vector.
 */
obs_vector_type *enkf_obs_get_vector(const enkf_obs_type *obs,
                                     const char *key) {
    return (obs_vector_type *)hash_get(obs->obs_hash, key);
}

obs_vector_type *enkf_obs_iget_vector(const enkf_obs_type *obs, int index) {
    return (obs_vector_type *)vector_iget(obs->obs_vector, index);
}

int enkf_obs_get_size(const enkf_obs_type *obs) {
    return vector_get_size(obs->obs_vector);
}

/**
   Adding inverse observation keys to the enkf_nodes; can be called
   several times.
*/
static void enkf_obs_update_keys(Cwrap<enkf_obs_type> enkf_obs) {
    /* First clear all existing observation keys. */
    for (auto &config_pair : enkf_obs->ensemble_config->config_nodes) {
        enkf_config_node_type *config_node = config_pair.second;
        stringlist_clear(config_node->obs_keys);
    }

    /* Add new observation keys. */
    hash_type *map = enkf_obs_alloc_data_map(enkf_obs);
    hash_iter_type *iter = hash_iter_alloc(map);
    const char *obs_key = hash_iter_get_next_key(iter);
    while (obs_key != NULL) {
        const char *state_kw = (const char *)hash_get(map, obs_key);

        enkf_config_node_type *node =
            enkf_obs->ensemble_config->config_nodes.at(state_kw);

        if (!stringlist_contains(node->obs_keys, obs_key))
            stringlist_append_copy(node->obs_keys, obs_key);

        obs_key = hash_iter_get_next_key(iter);
    }
    for (auto &[kw, node] : enkf_obs->ensemble_config->config_nodes) {
        stringlist_sort(node->obs_keys, NULL);
    }
    hash_iter_free(iter);
    hash_free(map);
}

/** Handle HISTORY_OBSERVATION instances. */
static void
handle_history_observation(Cwrap<enkf_obs_type> enkf_obs,
                           std::shared_ptr<conf_instance_type> enkf_conf,
                           double std_cutoff) {
    auto num_reports = enkf_obs->obs_time.size();
    std::vector<std::string> hist_obs_keys =
        conf_instance_alloc_list_of_sub_instances_of_class_by_name(
            enkf_conf, "HISTORY_OBSERVATION");
    int num_hist_obs = hist_obs_keys.size();

    if (num_hist_obs > 0 && enkf_obs->refcase == NULL) {
        throw exc::invalid_argument(
            "REFCASE is required for HISTORY_OBSERVATION");
    }

    for (int i = 0; i < num_hist_obs; i++) {
        std::string obs_key = hist_obs_keys[i];

        if (!enkf_obs->history) {
            fprintf(stderr,
                    "** Warning: no history object registered - observation:%s "
                    "is ignored\n",
                    obs_key.c_str());
            break;
        }
        auto hist_obs_conf =
            conf_instance_get_sub_instance_ref(enkf_conf, obs_key.c_str());

        enkf_config_node_type *config_node = ensemble_config_add_summary(
            enkf_obs->ensemble_config, obs_key.c_str(), LOAD_FAIL_WARN);

        enkf_obs->ensemble_config->summary_keys.push_back(obs_key);

        if (config_node == NULL) {
            fprintf(stderr,
                    "** Warning: summary:%s does not exist - observation:%s "
                    "not added.\n",
                    obs_key.c_str(), obs_key.c_str());
            break;
        }

        obs_vector_type *obs_vector =
            obs_vector_alloc(SUMMARY_OBS, obs_key.c_str(),
                             ensemble_config_get_node(enkf_obs->ensemble_config,
                                                      obs_key.c_str()),
                             num_reports);
        if (obs_vector != NULL) {
            if (obs_vector_load_from_HISTORY_OBSERVATION(
                    obs_vector, hist_obs_conf, enkf_obs->obs_time,
                    enkf_obs->history, std_cutoff, enkf_obs->refcase)) {
                enkf_obs_add_obs_vector(enkf_obs, obs_vector);
            } else {
                fprintf(stderr,
                        "** Could not load historical data for observation:%s "
                        "- ignored\n",
                        obs_key.c_str());

                obs_vector_free(obs_vector);
            }
        }
    }
}

/** Handle SUMMARY_OBSERVATION instances. */
static void
handle_summary_observation(Cwrap<enkf_obs_type> enkf_obs,
                           std::shared_ptr<conf_instance_type> enkf_conf) {
    auto num_reports = enkf_obs->obs_time.size();
    std::vector<std::string> sum_obs_keys =
        conf_instance_alloc_list_of_sub_instances_of_class_by_name(
            enkf_conf, "SUMMARY_OBSERVATION");
    const int num_sum_obs = sum_obs_keys.size();

    for (int i = 0; i < num_sum_obs; i++) {
        std::string obs_key = sum_obs_keys[i];
        auto sum_obs_conf =
            conf_instance_get_sub_instance_ref(enkf_conf, obs_key.c_str());
        const char *sum_key =
            conf_instance_get_item_value_ref(sum_obs_conf, "KEY");

        /* check if have sum_key exists */
        enkf_config_node_type *config_node = ensemble_config_add_summary(
            enkf_obs->ensemble_config, sum_key, LOAD_FAIL_WARN);

        enkf_obs->ensemble_config->summary_keys.push_back(sum_key);

        if (config_node == NULL) {
            fprintf(stderr,
                    "** Warning: summary key:%s does not exist - observation "
                    "key:%s not added.\n",
                    sum_key, obs_key.c_str());
            break;
        }

        /* Check if obs_vector is alloc'd */
        obs_vector_type *obs_vector = obs_vector_alloc(
            SUMMARY_OBS, obs_key.c_str(),
            ensemble_config_get_node(enkf_obs->ensemble_config, sum_key),
            num_reports);
        if (obs_vector == NULL)
            break;

        obs_vector_load_from_SUMMARY_OBSERVATION(obs_vector, sum_obs_conf,
                                                 enkf_obs->obs_time);
        enkf_obs_add_obs_vector(enkf_obs, obs_vector);
    }
}

/** Handle GENERAL_OBSERVATION instances. */
static void
handle_general_observation(Cwrap<enkf_obs_type> enkf_obs,
                           std::shared_ptr<conf_instance_type> enkf_conf) {
    auto obs_keys = conf_instance_alloc_list_of_sub_instances_of_class_by_name(
        enkf_conf, "GENERAL_OBSERVATION");
    int num_obs = obs_keys.size();

    for (int i = 0; i < num_obs; i++) {
        std::string obs_key = obs_keys[i];
        auto gen_obs_conf =
            conf_instance_get_sub_instance_ref(enkf_conf, obs_key.c_str());

        const char *state_kw =
            conf_instance_get_item_value_ref(gen_obs_conf, "DATA");
        obs_vector_type *obs_vector = NULL;
        if (!ensemble_config_has_key(enkf_obs->ensemble_config, state_kw)) {

            fprintf(stderr,
                    "** Warning the ensemble key:%s does not exist - "
                    "observation:%s not added \n",
                    state_kw, obs_key.c_str());
        } else {
            enkf_config_node_type *config_node =
                ensemble_config_get_node(enkf_obs->ensemble_config, state_kw);
            obs_vector = obs_vector_alloc_from_GENERAL_OBSERVATION(
                gen_obs_conf, enkf_obs->obs_time, config_node);
        }
        if (obs_vector != NULL)
            enkf_obs_add_obs_vector(enkf_obs, obs_vector);
    }
}

std::shared_ptr<conf_class_type> enkf_obs_get_obs_conf_class() {
    const char *enkf_conf_help =
        "An instance of the class ENKF_CONFIG shall contain neccessary "
        "infomation to run the enkf.";
    auto enkf_conf_class =
        make_conf_class("ENKF_CONFIG", true, false, enkf_conf_help);
    conf_class_set_help(enkf_conf_class, enkf_conf_help);

    /* Create and insert HISTORY_OBSERVATION class. */
    {
        const char *help_class_history_observation =
            "The class HISTORY_OBSERVATION is used to condition on a time "
            "series from the production history. The name of the an "
            "instance "
            "is used to define the item to condition on, and should be in "
            "summary.x syntax. E.g., creating a HISTORY_OBSERVATION "
            "instance "
            "with name GOPR:P4 conditions on GOPR for group P4.";

        auto history_observation_class =
            make_conf_class("HISTORY_OBSERVATION", false, false,
                            help_class_history_observation);

        auto item_spec_error_mode =
            make_conf_item_spec("ERROR_MODE", true, DT_STR,
                                "The string ERROR_MODE gives the error "
                                "mode for the observation.");
        conf_item_spec_add_restriction(item_spec_error_mode, "REL");
        conf_item_spec_add_restriction(item_spec_error_mode, "ABS");
        conf_item_spec_add_restriction(item_spec_error_mode, "RELMIN");
        conf_item_spec_set_default_value(item_spec_error_mode, "RELMIN");

        auto item_spec_error = make_conf_item_spec(
            "ERROR", true, DT_POSFLOAT,
            "The positive floating number ERROR gives the standard "
            "deviation "
            "(ABS) or the relative uncertainty (REL/RELMIN) of the "
            "observations.");
        conf_item_spec_set_default_value(item_spec_error, "0.10");

        auto item_spec_error_min =
            make_conf_item_spec("ERROR_MIN", true, DT_POSFLOAT,
                                "The positive floating point number "
                                "ERROR_MIN gives the minimum "
                                "value for the standard deviation of the "
                                "observation when RELMIN "
                                "is used.");
        conf_item_spec_set_default_value(item_spec_error_min, "0.10");

        conf_class_insert_item_spec(history_observation_class,
                                    item_spec_error_mode);
        conf_class_insert_item_spec(history_observation_class, item_spec_error);
        conf_class_insert_item_spec(history_observation_class,
                                    item_spec_error_min);

        /* Sub class segment. */
        {
            const char *help_class_segment =
                "The class SEGMENT is used to fine tune the error model.";
            auto segment_class =
                make_conf_class("SEGMENT", false, false, help_class_segment);

            auto item_spec_start_segment = make_conf_item_spec(
                "START", true, DT_INT, "The first restart in the segment.");
            auto item_spec_stop_segment = make_conf_item_spec(
                "STOP", true, DT_INT, "The last restart in the segment.");

            auto item_spec_error_mode_segment =
                make_conf_item_spec("ERROR_MODE", true, DT_STR,
                                    "The string ERROR_MODE gives the error "
                                    "mode for the observation.");
            conf_item_spec_add_restriction(item_spec_error_mode_segment, "REL");
            conf_item_spec_add_restriction(item_spec_error_mode_segment, "ABS");
            conf_item_spec_add_restriction(item_spec_error_mode_segment,
                                           "RELMIN");
            conf_item_spec_set_default_value(item_spec_error_mode_segment,
                                             "RELMIN");

            auto item_spec_error_segment = make_conf_item_spec(
                "ERROR", true, DT_POSFLOAT,
                "The positive floating number ERROR gives the standard "
                "deviation (ABS) or the relative uncertainty "
                "(REL/RELMIN) of "
                "the observations.");
            conf_item_spec_set_default_value(item_spec_error_segment, "0.10");

            auto item_spec_error_min_segment = make_conf_item_spec(
                "ERROR_MIN", true, DT_POSFLOAT,
                "The positive floating point number ERROR_MIN gives "
                "the "
                "minimum value for the standard deviation of the "
                "observation when RELMIN is used.");
            conf_item_spec_set_default_value(item_spec_error_min_segment,
                                             "0.10");

            conf_class_insert_item_spec(segment_class, item_spec_start_segment);
            conf_class_insert_item_spec(segment_class, item_spec_stop_segment);
            conf_class_insert_item_spec(segment_class,
                                        item_spec_error_mode_segment);
            conf_class_insert_item_spec(segment_class, item_spec_error_segment);
            conf_class_insert_item_spec(segment_class,
                                        item_spec_error_min_segment);

            conf_class_insert_sub_class(history_observation_class,
                                        segment_class);
        }

        conf_class_insert_sub_class(enkf_conf_class, history_observation_class);
    }

    /* Create and insert SUMMARY_OBSERVATION class. */
    {
        const char *help_class_summary_observation =
            "The class SUMMARY_OBSERVATION can be used to condition on any "
            "observation whos simulated value is written to the summary "
            "file.";
        auto summary_observation_class =
            make_conf_class("SUMMARY_OBSERVATION", false, false,
                            help_class_summary_observation);

        const char *help_item_spec_value =
            "The floating point number VALUE gives the observed value.";
        auto item_spec_value =
            make_conf_item_spec("VALUE", true, DT_FLOAT, help_item_spec_value);

        const char *help_item_spec_error =
            "The positive floating point number ERROR is the standard "
            "deviation of the observed value.";
        auto item_spec_error = make_conf_item_spec("ERROR", true, DT_POSFLOAT,
                                                   help_item_spec_error);

        const char *help_item_spec_date =
            "The DATE item gives the observation time as the date date it "
            "occured. Format is YYYY-MM-DD.";
        auto item_spec_date =
            make_conf_item_spec("DATE", false, DT_DATE, help_item_spec_date);

        const char *help_item_spec_days =
            "The DAYS item gives the observation time as days after "
            "simulation "
            "start.";
        auto item_spec_days = make_conf_item_spec("DAYS", false, DT_POSFLOAT,
                                                  help_item_spec_days);

        const char *help_item_spec_hours =
            "The HOURS item gives the observation time as hours after "
            "simulation start.";
        auto item_spec_hours = make_conf_item_spec("HOURS", false, DT_POSFLOAT,
                                                   help_item_spec_hours);

        const char *help_item_spec_restart =
            "The RESTART item gives the observation time as the ECLIPSE "
            "restart nr.";
        auto item_spec_restart = make_conf_item_spec(
            "RESTART", false, DT_POSINT, help_item_spec_restart);

        const char *help_item_spec_sumkey =
            "The string SUMMARY_KEY is used to look up the simulated value "
            "in "
            "the summary file. It has the same format as the summary.x "
            "program, e.g. WOPR:P4";
        auto item_spec_sumkey =
            make_conf_item_spec("KEY", true, DT_STR, help_item_spec_sumkey);

        auto item_spec_error_min =
            make_conf_item_spec("ERROR_MIN", true, DT_POSFLOAT,
                                "The positive floating point number "
                                "ERROR_MIN gives the minimum "
                                "value for the standard deviation of the "
                                "observation when RELMIN "
                                "is used.");
        auto item_spec_error_mode =
            make_conf_item_spec("ERROR_MODE", true, DT_STR,
                                "The string ERROR_MODE gives the error "
                                "mode for the observation.");

        conf_item_spec_add_restriction(item_spec_error_mode, "REL");
        conf_item_spec_add_restriction(item_spec_error_mode, "ABS");
        conf_item_spec_add_restriction(item_spec_error_mode, "RELMIN");
        conf_item_spec_set_default_value(item_spec_error_mode, "ABS");
        conf_item_spec_set_default_value(item_spec_error_min, "0.10");

        conf_class_insert_item_spec(summary_observation_class, item_spec_value);
        conf_class_insert_item_spec(summary_observation_class, item_spec_error);
        conf_class_insert_item_spec(summary_observation_class, item_spec_date);
        conf_class_insert_item_spec(summary_observation_class, item_spec_days);
        conf_class_insert_item_spec(summary_observation_class, item_spec_hours);
        conf_class_insert_item_spec(summary_observation_class,
                                    item_spec_restart);
        conf_class_insert_item_spec(summary_observation_class,
                                    item_spec_sumkey);
        conf_class_insert_item_spec(summary_observation_class,
                                    item_spec_error_mode);
        conf_class_insert_item_spec(summary_observation_class,
                                    item_spec_error_min);

        /* Create a mutex on DATE, DAYS and RESTART. */
        auto time_mutex =
            conf_class_new_item_mutex(summary_observation_class, true, false);

        conf_item_mutex_add_item_spec(time_mutex, item_spec_date);
        conf_item_mutex_add_item_spec(time_mutex, item_spec_days);
        conf_item_mutex_add_item_spec(time_mutex, item_spec_hours);
        conf_item_mutex_add_item_spec(time_mutex, item_spec_restart);
        conf_item_mutex_add_item_spec(time_mutex, item_spec_days);

        conf_class_insert_sub_class(enkf_conf_class, summary_observation_class);
    }

    /* Create and insert class for general observations. */
    {
        const char *help_item_spec_restart =
            "The RESTART item gives the observation time as the ECLIPSE "
            "restart nr.";
        const char *help_item_spec_field =
            "The item DATA gives the observed GEN_DATA instance.";
        const char *help_item_spec_date =
            "The DATE item gives the observation time as the date date it "
            "occured. Format is YYYY-MM-DD.";
        const char *help_item_spec_days =
            "The DAYS item gives the observation time as days after "
            "simulation "
            "start.";
        const char *help_item_spec_hours =
            "The HOURS item gives the observation time as hours after "
            "simulation start.";

        auto gen_obs_class =
            make_conf_class("GENERAL_OBSERVATION", false, false,
                            "The class general_observation is used "
                            "for general observations");

        auto item_spec_field =
            make_conf_item_spec("DATA", true, DT_STR, help_item_spec_field);
        auto item_spec_date =
            make_conf_item_spec("DATE", false, DT_DATE, help_item_spec_date);
        auto item_spec_days = make_conf_item_spec("DAYS", false, DT_POSFLOAT,
                                                  help_item_spec_days);
        auto item_spec_hours = make_conf_item_spec("HOURS", false, DT_POSFLOAT,
                                                   help_item_spec_hours);
        auto item_spec_restart = make_conf_item_spec("RESTART", false, DT_INT,
                                                     help_item_spec_restart);

        conf_class_insert_item_spec(gen_obs_class, item_spec_field);
        conf_class_insert_item_spec(gen_obs_class, item_spec_date);
        conf_class_insert_item_spec(gen_obs_class, item_spec_days);
        conf_class_insert_item_spec(gen_obs_class, item_spec_hours);
        conf_class_insert_item_spec(gen_obs_class, item_spec_restart);
        /* Create a mutex on DATE, DAYS and RESTART. */
        {
            auto time_mutex =
                conf_class_new_item_mutex(gen_obs_class, true, false);

            conf_item_mutex_add_item_spec(time_mutex, item_spec_date);
            conf_item_mutex_add_item_spec(time_mutex, item_spec_days);
            conf_item_mutex_add_item_spec(time_mutex, item_spec_hours);
            conf_item_mutex_add_item_spec(time_mutex, item_spec_restart);
        }

        {
            auto item_spec_obs_file = make_conf_item_spec(
                "OBS_FILE", false, DT_FILE,
                "The name of an (ascii) file with observation values.");
            auto item_spec_value = make_conf_item_spec(
                "VALUE", false, DT_FLOAT, "One scalar observation value.");
            auto item_spec_error = make_conf_item_spec(
                "ERROR", false, DT_FLOAT, "One scalar observation error.");
            auto value_mutex =
                conf_class_new_item_mutex(gen_obs_class, true, false);
            auto value_error_mutex =
                conf_class_new_item_mutex(gen_obs_class, false, true);

            conf_class_insert_item_spec(gen_obs_class, item_spec_obs_file);
            conf_class_insert_item_spec(gen_obs_class, item_spec_value);
            conf_class_insert_item_spec(gen_obs_class, item_spec_error);

            /* If the observation is in terms of VALUE - we must also have ERROR.
         The conf system does not (currently ??) enforce this dependency. */

            conf_item_mutex_add_item_spec(value_mutex, item_spec_value);
            conf_item_mutex_add_item_spec(value_mutex, item_spec_obs_file);

            conf_item_mutex_add_item_spec(value_error_mutex, item_spec_value);
            conf_item_mutex_add_item_spec(value_error_mutex, item_spec_error);
        }

        /*
       The default is that all the elements in DATA are observed, but
       we can restrict ourselves to a list of indices, with either the
       INDEX_LIST or INDEX_FILE keywords.
    */
        {
            auto item_spec_index_list =
                make_conf_item_spec("INDEX_LIST", false, DT_STR,
                                    "A list of indicies - possibly with "
                                    "ranges which should be "
                                    "observed in the target field.");
            auto item_spec_index_file =
                make_conf_item_spec("INDEX_FILE", false, DT_FILE,
                                    "An ASCII file containing a list of "
                                    "indices which should be "
                                    "observed in the target field.");
            auto index_mutex =
                conf_class_new_item_mutex(gen_obs_class, false, false);

            conf_class_insert_item_spec(gen_obs_class, item_spec_index_list);
            conf_class_insert_item_spec(gen_obs_class, item_spec_index_file);
            conf_item_mutex_add_item_spec(index_mutex, item_spec_index_list);
            conf_item_mutex_add_item_spec(index_mutex, item_spec_index_file);
        }

        conf_class_insert_sub_class(enkf_conf_class, gen_obs_class);
    }

    return enkf_conf_class;
}

/**
   Allocates a stringlist of obs target keys which correspond to
   summary observations, these are then added to the state vector in
   enkf_main.
*/
stringlist_type *enkf_obs_alloc_typed_keylist(enkf_obs_type *enkf_obs,
                                              obs_impl_type obs_type) {
    stringlist_type *vars = stringlist_alloc_new();
    hash_iter_type *iter = hash_iter_alloc(enkf_obs->obs_hash);
    const char *key = hash_iter_get_next_key(iter);
    while (key != NULL) {
        obs_vector_type *obs_vector =
            (obs_vector_type *)hash_get(enkf_obs->obs_hash, key);
        if (obs_vector_get_impl_type(obs_vector) == obs_type)
            stringlist_append_copy(vars, key);
        key = hash_iter_get_next_key(iter);
    }
    hash_iter_free(iter);
    return vars;
}

obs_impl_type enkf_obs_get_type(const enkf_obs_type *enkf_obs,
                                const char *key) {
    obs_vector_type *obs_vector =
        (obs_vector_type *)hash_get(enkf_obs->obs_hash, key);
    return obs_vector_get_impl_type(obs_vector);
}

stringlist_type *enkf_obs_alloc_matching_keylist(const enkf_obs_type *enkf_obs,
                                                 const char *input_string) {

    stringlist_type *obs_keys = hash_alloc_stringlist(enkf_obs->obs_hash);

    if (!input_string)
        return obs_keys;

    stringlist_type *matching_keys = stringlist_alloc_new();
    int obs_keys_count = stringlist_get_size(obs_keys);

    ert::split(input_string, ' ',
               [&obs_keys, &matching_keys, &obs_keys_count](auto input_key) {
                   for (int j = 0; j < obs_keys_count; j++) {
                       const char *obs_key = stringlist_iget(obs_keys, j);

                       if (util_string_match(obs_key,
                                             std::string(input_key).c_str()) &&
                           !stringlist_contains(matching_keys, obs_key))
                           stringlist_append_copy(matching_keys, obs_key);
                   }
               });

    stringlist_sort(matching_keys, nullptr);

    return matching_keys;
}

/**
   @brief returns a map from the observation keys to the observed state keys.

   This function allocates a hash table which looks like this:

     {"OBS_KEY1": "STATE_KEY1", "OBS_KEY2": "STATE_KEY2", "OBS_KEY3": "STATE_KEY3", ....}

   where "OBS_KEY" represents the keys in the enkf_obs hash, and the
   value they are pointing at are the enkf_state keywords they are
   measuring. For instance if we have an observation with key "RFT_1A"
   the entry in the table will be:  ... "RFT_1A":  "PRESSURE", ..
   since an RFT observation observes the pressure.

   Let us consider the watercut in a well. Then the state_kw will
   typically be WWCT:P1 for a well named 'P1'. Let us assume that this
   well is observed both as a normal HISTORY observation from
   SCHEDULE, and from two separator tests, called S1 and S2. Then the
   hash table will look like this:

       "WWCT:P1": "WWCT:P1",
       "S1"     : "WWCT:P1",
       "S2"     : "WWCT:P1"


   I.e. there are three different observations keys, all observing the
   same state_kw.
*/
hash_type *enkf_obs_alloc_data_map(enkf_obs_type *enkf_obs) {
    hash_type *map = hash_alloc();
    hash_iter_type *iter = hash_iter_alloc(enkf_obs->obs_hash);
    const char *key = hash_iter_get_next_key(iter);
    while (key != NULL) {
        obs_vector_type *obs_vector =
            (obs_vector_type *)hash_get(enkf_obs->obs_hash, key);
        hash_insert_ref(map, key, obs_vector_get_state_kw(obs_vector));
        key = hash_iter_get_next_key(iter);
    }
    hash_iter_free(iter);
    return map;
}

hash_iter_type *enkf_obs_alloc_iter(const enkf_obs_type *enkf_obs) {
    return hash_iter_alloc(enkf_obs->obs_hash);
}

namespace {
py::handle pybind_alloc(int history_,
                        std::shared_ptr<TimeMap> external_time_map,
                        Cwrap<ecl_sum_type> refcase,
                        Cwrap<ensemble_config_type> ensemble_config) {
    auto history = static_cast<history_source_type>(history_);
    auto ptr =
        enkf_obs_alloc(history, external_time_map, refcase, ensemble_config);
    return PyLong_FromVoidPtr(ptr);
}

} // namespace

ERT_CLIB_SUBMODULE("enkf_obs", m) {
    using namespace py::literals;

    py::class_<conf_instance_type, std::shared_ptr<conf_instance_type>>(
        m, "ConfInstance")
        .def(py::init([](std::string config_file) {
                 auto enkf_conf_class = enkf_obs_get_obs_conf_class();
                 auto enkf_conf = conf_instance_alloc_from_file(
                     enkf_conf_class, "enkf_conf", config_file.c_str());
                 return enkf_conf;
             }),
             "config_file"_a)
        .def("get_errors", [](std::shared_ptr<conf_instance_type> self) {
            auto errors = conf_instance_get_path_error(self);
            if (errors) {
                return std::string(
                           "The following keywords in your configuration did "
                           "not resolve to a valid path:\n ") +
                       errors;
            }
            if (!conf_instance_validate(self))
                return std::string("Error in observations configuration file");
            return std::string("");
        });
    m.def("alloc", pybind_alloc, "history"_a, "time_map"_a, "refcase"_a,
          "ensemble_config"_a);
    m.def("handle_history_observation", handle_history_observation,
          py::arg("self"), py::arg("conf_instance"), py::arg("std_cutoff"));
    m.def("handle_summary_observation", handle_summary_observation,
          py::arg("self"), py::arg("conf_instance"));
    m.def("handle_general_observation", handle_general_observation,
          py::arg("self"), py::arg("conf_instance"));
    m.def("update_keys", enkf_obs_update_keys, py::arg("self"));
}
