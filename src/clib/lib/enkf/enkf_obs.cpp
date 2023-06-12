#include <cmath>
#include <cppitertools/enumerate.hpp>
#include <utility>

#include <ert/config/conf.hpp>
#include <ert/ecl/ecl_grid.h>
#include <ert/ecl/ecl_sum.h>
#include <ert/util/hash.h>
#include <ert/util/type_vector_functions.h>
#include <ert/util/vector.h>

#include <ert/config/conf.hpp>
#include <ert/ecl/ecl_grid.h>
#include <ert/ecl/ecl_sum.h>
#include <ert/except.hpp>
#include <ert/python.hpp>
#include <ert/res_util/string.hpp>
#include <ert/util/hash.h>
#include <ert/util/type_vector_functions.h>
#include <ert/util/vector.h>
#include <map>

std::shared_ptr<conf_class> enkf_obs_get_obs_conf_class() {
    const char *enkf_conf_help =
        "An instance of the class ENKF_CONFIG shall contain neccessary "
        "infomation to run the enkf.";
    auto enkf_conf_class = std::make_shared<conf_class>("ENKF_CONFIG", true,
                                                        false, enkf_conf_help);

    /* Create and insert HISTORY_OBSERVATION class. */
    {
        std::string help_class_history_observation =
            "The class HISTORY_OBSERVATION is used to condition on a time "
            "series from the production history. The name of the an "
            "instance "
            "is used to define the item to condition on, and should be in "
            "summary.x syntax. E.g., creating a HISTORY_OBSERVATION "
            "instance "
            "with name GOPR:P4 conditions on GOPR for group P4.";

        auto history_observation_class =
            std::make_shared<conf_class>("HISTORY_OBSERVATION", false, false,
                                         help_class_history_observation);

        auto item_spec_error_mode = std::make_shared<conf_item_spec>(
            "ERROR_MODE", true, DT_STR,
            "The string ERROR_MODE gives the error "
            "mode for the observation.");
        item_spec_error_mode->add_restriction("REL");
        item_spec_error_mode->add_restriction("ABS");
        item_spec_error_mode->add_restriction("RELMIN");
        item_spec_error_mode->default_value = "RELMIN";

        auto item_spec_error = std::make_shared<conf_item_spec>(
            "ERROR", true, DT_POSFLOAT,
            "The positive floating number ERROR gives the standard "
            "deviation "
            "(ABS) or the relative uncertainty (REL/RELMIN) of the "
            "observations.");
        item_spec_error->default_value = "0.10";

        auto item_spec_error_min = std::make_shared<conf_item_spec>(
            "ERROR_MIN", true, DT_POSFLOAT,
            "The positive floating point number "
            "ERROR_MIN gives the minimum "
            "value for the standard deviation of the "
            "observation when RELMIN "
            "is used.");
        item_spec_error_min->default_value = "0.10";

        history_observation_class->insert_item_spec(item_spec_error_mode);
        history_observation_class->insert_item_spec(item_spec_error);
        history_observation_class->insert_item_spec(item_spec_error_min);

        std::string help_class_segment =
            "The class SEGMENT is used to fine tune the error model.";
        auto segment_class = std::make_shared<conf_class>(
            "SEGMENT", false, false, help_class_segment);

        auto item_spec_start_segment = std::make_shared<conf_item_spec>(
            "START", true, DT_INT, "The first restart in the segment.");
        auto item_spec_stop_segment = std::make_shared<conf_item_spec>(
            "STOP", true, DT_INT, "The last restart in the segment.");

        auto item_spec_error_mode_segment = std::make_shared<conf_item_spec>(
            "ERROR_MODE", true, DT_STR,
            "The string ERROR_MODE gives the error "
            "mode for the observation.");
        item_spec_error_mode_segment->add_restriction("REL");
        item_spec_error_mode_segment->add_restriction("ABS");
        item_spec_error_mode_segment->add_restriction("RELMIN");
        item_spec_error_mode_segment->default_value = "RELMIN";

        auto item_spec_error_segment = std::make_shared<conf_item_spec>(
            "ERROR", true, DT_POSFLOAT,
            "The positive floating number ERROR gives the standard "
            "deviation (ABS) or the relative uncertainty "
            "(REL/RELMIN) of "
            "the observations.");
        item_spec_error_segment->default_value = "0.10";

        auto item_spec_error_min_segment = std::make_shared<conf_item_spec>(
            "ERROR_MIN", true, DT_POSFLOAT,
            "The positive floating point number ERROR_MIN gives "
            "the "
            "minimum value for the standard deviation of the "
            "observation when RELMIN is used.");
        item_spec_error_min_segment->default_value = "0.10";

        segment_class->insert_item_spec(item_spec_start_segment);
        segment_class->insert_item_spec(item_spec_stop_segment);
        segment_class->insert_item_spec(item_spec_error_mode_segment);
        segment_class->insert_item_spec(item_spec_error_segment);
        segment_class->insert_item_spec(item_spec_error_min_segment);

        history_observation_class->insert_sub_class(segment_class);

        enkf_conf_class->insert_sub_class(history_observation_class);
    }

    /* Create and insert SUMMARY_OBSERVATION class. */
    {
        const char *help_class_summary_observation =
            "The class SUMMARY_OBSERVATION can be used to condition on any "
            "observation whos simulated value is written to the summary "
            "file.";
        auto summary_observation_class =
            std::make_shared<conf_class>("SUMMARY_OBSERVATION", false, false,
                                         help_class_summary_observation);

        const char *help_item_spec_value =
            "The floating point number VALUE gives the observed value.";
        auto item_spec_value = std::make_shared<conf_item_spec>(
            "VALUE", true, DT_FLOAT, help_item_spec_value);

        const char *help_item_spec_error =
            "The positive floating point number ERROR is the standard "
            "deviation of the observed value.";
        auto item_spec_error = std::make_shared<conf_item_spec>(
            "ERROR", true, DT_POSFLOAT, help_item_spec_error);

        const char *help_item_spec_date =
            "The DATE item gives the observation time as the date date it "
            "occured. Format is YYYY-MM-DD.";
        auto item_spec_date = std::make_shared<conf_item_spec>(
            "DATE", false, DT_DATE, help_item_spec_date);

        const char *help_item_spec_days =
            "The DAYS item gives the observation time as days after "
            "simulation "
            "start.";
        auto item_spec_days = std::make_shared<conf_item_spec>(
            "DAYS", false, DT_POSFLOAT, help_item_spec_days);

        const char *help_item_spec_hours =
            "The HOURS item gives the observation time as hours after "
            "simulation start.";
        auto item_spec_hours = std::make_shared<conf_item_spec>(
            "HOURS", false, DT_POSFLOAT, help_item_spec_hours);

        const char *help_item_spec_restart =
            "The RESTART item gives the observation time as the ECLIPSE "
            "restart nr.";
        auto item_spec_restart = std::make_shared<conf_item_spec>(
            "RESTART", false, DT_POSINT, help_item_spec_restart);

        const char *help_item_spec_sumkey =
            "The string SUMMARY_KEY is used to look up the simulated value "
            "in "
            "the summary file. It has the same format as the summary.x "
            "program, e.g. WOPR:P4";
        auto item_spec_sumkey = std::make_shared<conf_item_spec>(
            "KEY", true, DT_STR, help_item_spec_sumkey);

        auto item_spec_error_min = std::make_shared<conf_item_spec>(
            "ERROR_MIN", true, DT_POSFLOAT,
            "The positive floating point number "
            "ERROR_MIN gives the minimum "
            "value for the standard deviation of the "
            "observation when RELMIN "
            "is used.");
        auto item_spec_error_mode = std::make_shared<conf_item_spec>(
            "ERROR_MODE", true, DT_STR,
            "The string ERROR_MODE gives the error "
            "mode for the observation.");

        item_spec_error_mode->add_restriction("REL");
        item_spec_error_mode->add_restriction("ABS");
        item_spec_error_mode->add_restriction("RELMIN");
        item_spec_error_mode->default_value = "ABS";
        item_spec_error_min->default_value = "0.10";

        summary_observation_class->insert_item_spec(item_spec_value);
        summary_observation_class->insert_item_spec(item_spec_error);
        summary_observation_class->insert_item_spec(item_spec_date);
        summary_observation_class->insert_item_spec(item_spec_days);
        summary_observation_class->insert_item_spec(item_spec_hours);
        summary_observation_class->insert_item_spec(item_spec_restart);
        summary_observation_class->insert_item_spec(item_spec_sumkey);
        summary_observation_class->insert_item_spec(item_spec_error_mode);
        summary_observation_class->insert_item_spec(item_spec_error_min);

        /* Create a mutex on DATE, DAYS and RESTART. */
        auto time_mutex =
            summary_observation_class->new_item_mutex(true, false);

        time_mutex->add_item_spec(item_spec_date);
        time_mutex->add_item_spec(item_spec_days);
        time_mutex->add_item_spec(item_spec_hours);
        time_mutex->add_item_spec(item_spec_restart);
        time_mutex->add_item_spec(item_spec_days);

        enkf_conf_class->insert_sub_class(summary_observation_class);
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

        auto gen_obs_class = std::make_shared<conf_class>(
            "GENERAL_OBSERVATION", false, false,
            "The class general_observation is used "
            "for general observations");

        auto item_spec_field = std::make_shared<conf_item_spec>(
            "DATA", true, DT_STR, help_item_spec_field);
        auto item_spec_date = std::make_shared<conf_item_spec>(
            "DATE", false, DT_DATE, help_item_spec_date);
        auto item_spec_days = std::make_shared<conf_item_spec>(
            "DAYS", false, DT_POSFLOAT, help_item_spec_days);
        auto item_spec_hours = std::make_shared<conf_item_spec>(
            "HOURS", false, DT_POSFLOAT, help_item_spec_hours);
        auto item_spec_restart = std::make_shared<conf_item_spec>(
            "RESTART", false, DT_INT, help_item_spec_restart);

        gen_obs_class->insert_item_spec(item_spec_field);
        gen_obs_class->insert_item_spec(item_spec_date);
        gen_obs_class->insert_item_spec(item_spec_days);
        gen_obs_class->insert_item_spec(item_spec_hours);
        gen_obs_class->insert_item_spec(item_spec_restart);
        /* Create a mutex on DATE, DAYS and RESTART. */
        {
            auto time_mutex = gen_obs_class->new_item_mutex(true, false);

            time_mutex->add_item_spec(item_spec_date);
            time_mutex->add_item_spec(item_spec_days);
            time_mutex->add_item_spec(item_spec_hours);
            time_mutex->add_item_spec(item_spec_restart);
        }

        {
            auto item_spec_obs_file = std::make_shared<conf_item_spec>(
                "OBS_FILE", false, DT_FILE,
                "The name of an (ascii) file with observation values.");
            auto item_spec_value = std::make_shared<conf_item_spec>(
                "VALUE", false, DT_FLOAT, "One scalar observation value.");
            auto item_spec_error = std::make_shared<conf_item_spec>(
                "ERROR", false, DT_FLOAT, "One scalar observation error.");
            auto value_mutex = gen_obs_class->new_item_mutex(true, false);
            auto value_error_mutex = gen_obs_class->new_item_mutex(false, true);

            gen_obs_class->insert_item_spec(item_spec_obs_file);
            gen_obs_class->insert_item_spec(item_spec_value);
            gen_obs_class->insert_item_spec(item_spec_error);

            /* If the observation is in terms of VALUE - we must also have ERROR.
         The conf system does not (currently ??) enforce this dependency. */

            value_mutex->add_item_spec(item_spec_value);
            value_mutex->add_item_spec(item_spec_obs_file);

            value_error_mutex->add_item_spec(item_spec_value);
            value_error_mutex->add_item_spec(item_spec_error);
        }

        /*
       The default is that all the elements in DATA are observed, but
       we can restrict ourselves to a list of indices, with either the
       INDEX_LIST or INDEX_FILE keywords.
    */
        {
            auto item_spec_index_list = std::make_shared<conf_item_spec>(
                "INDEX_LIST", false, DT_STR,
                "A list of indicies - possibly with "
                "ranges which should be "
                "observed in the target field.");
            auto item_spec_index_file = std::make_shared<conf_item_spec>(
                "INDEX_FILE", false, DT_FILE,
                "An ASCII file containing a list of "
                "indices which should be "
                "observed in the target field.");
            auto index_mutex = gen_obs_class->new_item_mutex(false, false);

            gen_obs_class->insert_item_spec(item_spec_index_list);
            gen_obs_class->insert_item_spec(item_spec_index_file);
            index_mutex->add_item_spec(item_spec_index_list);
            index_mutex->add_item_spec(item_spec_index_file);
        }

        enkf_conf_class->insert_sub_class(gen_obs_class);
    }

    return enkf_conf_class;
}

ERT_CLIB_SUBMODULE("enkf_obs", m) {
    using namespace py::literals;

    py::class_<conf_instance, std::shared_ptr<conf_instance>>(m, "ConfInstance")
        .def(py::init([](std::string config_file) {
                 auto enkf_conf_class = enkf_obs_get_obs_conf_class();
                 auto enkf_conf = conf_instance::from_file(
                     enkf_conf_class, "enkf_conf", config_file);
                 return enkf_conf;
             }),
             "config_file"_a)
        .def("get_sub_instances",
             [](std::shared_ptr<conf_instance> self, std::string name) {
                 return self->get_sub_instances(name);
             })
        .def("__contains__",
             [](std::shared_ptr<conf_instance> self, std::string name) {
                 return self->has_value(name);
             })
        .def("__getitem__",
             [](std::shared_ptr<conf_instance> self, std::string name) {
                 return self->get_value(name);
             })
        .def_property_readonly(
            "name",
            [](std::shared_ptr<conf_instance> self) { return self->name; })
        .def("get_errors", [](std::shared_ptr<conf_instance> self) {
            return self->validate();
        });
    m.def("read_from_refcase",
          [](Cwrap<ecl_sum_type> refcase, std::string local_key) {
              int num_steps = ecl_sum_get_last_report_step(refcase);
              std::vector<bool> valid(num_steps + 1);
              std::vector<double> value(num_steps + 1);
              for (int tstep = 0; tstep <= num_steps; tstep++) {
                  if (ecl_sum_has_report_step(refcase, tstep)) {
                      int time_index = ecl_sum_iget_report_end(refcase, tstep);
                      value[tstep] = ecl_sum_get_general_var(
                          refcase, time_index, local_key.c_str());
                      valid[tstep] = true;
                  } else {
                      valid[tstep] = false;
                  }
              }

              return std::make_pair(valid, value);
          });
}
