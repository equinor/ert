/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'rng_config.c' is part of ERT - Ensemble based Reservoir Tool.

   ERT is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   ERT is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or
   FITNESS FOR A PARTICULAR PURPOSE.

   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
   for more details.
*/

#include <stdlib.h>

#include <ert/config/config_parser.hpp>
#include <ert/python.hpp>
#include <ert/util/rng.h>
#include <ert/util/util.h>

#include <ert/logging.hpp>
#include <string>

#include <ert/enkf/config_keys.hpp>
#include <ert/enkf/model_config.hpp>
#include <ert/enkf/rng_config.hpp>

static auto logger = ert::get_logger("enkf");

struct rng_config_struct {
    char *random_seed;
};

const char *rng_config_get_random_seed(const rng_config_type *rng_config) {
    return rng_config->random_seed;
}

rng_config_type *rng_config_alloc(const char *random_seed) {
    auto rng_config = (rng_config_type *)malloc(sizeof(rng_config_type *));
    rng_config->random_seed = util_alloc_string_copy(random_seed);
    return rng_config;
}

void rng_config_free(rng_config_type *rng) {
    free(rng->random_seed);
    free(rng);
}

void rng_config_add_config_items(config_parser_type *parser) {
    config_add_key_value(parser, RANDOM_SEED_KEY, false, CONFIG_STRING);
}

ERT_CLIB_SUBMODULE("rng_config", m) {
    using namespace py::literals;
    m.def("log_seed", [](py::object rng_) {
        auto rng = ert::from_cwrap<rng_type>(rng_);
        unsigned int random_seed[4];
        rng_get_state(rng, (char *)random_seed);

        char random_seed_str[10 * 4 + 1];
        random_seed_str[0] = '\0';
        char *uint_fmt = util_alloc_sprintf("%%0%du", 10);

        for (int i = 0; i < 4; ++i) {
            char *elem = util_alloc_sprintf(uint_fmt, random_seed[i]);
            strcat(random_seed_str, elem);
            free(elem);
        }
        free(uint_fmt);
        logger->info(
            "To repeat this experiment, add the following random seed to "
            "your config file:");
        logger->info("RANDOM_SEED {}", random_seed_str);
    });
}
