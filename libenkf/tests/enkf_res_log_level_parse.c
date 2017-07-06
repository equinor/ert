/*
   Copyright (C) 2017  Statoil ASA, Norway.

   The file 'enkf_res_log_level_parse.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <ert/util/test_util.h>
#include <ert/util/string_util.h>
#include <ert/enkf/enkf_main.h>

typedef struct {
  const char * log_keyword; // keyword written in the config file
  const char * log_old_numeric_str; // *old* integer value (as a string)
  const message_level_type log_enum; // enum for the new log-level
} log_tuple;
log_tuple log_levels[] = {
        {"CRITICAL", "0", LOG_CRITICAL},
        {"ERROR", "1", LOG_ERROR},
        {"WARNING", "2", LOG_WARNING},
        {"INFO", "3", LOG_INFO},
        {"DEBUG", "4", LOG_DEBUG}};

/* Old numeric values are parsed and converted properly */
void test_parse_old_numeric_positive() {
  for (int i = 0; i < 5; i++) {
    log_tuple curr_log_tuple = log_levels[i];
    test_assert_int_equal(res_log_level_parser(curr_log_tuple.log_old_numeric_str), curr_log_tuple.log_enum);
  }
}

void test_parse_keywords_positive() {
  for (int i = 0; i < 5; i++) {
    test_assert_int_equal(res_log_level_parser(log_levels[i].log_keyword), log_levels[i].log_enum);
  }
}

void test_parse_negative_becomes_default() {
  for (int i = 0; i < 5; i++) {
    test_assert_int_equal(res_log_level_parser(log_levels[i].log_keyword), log_levels[i].log_enum);
  }
}

int main( int argc , char ** argv) {
  test_parse_old_numeric_positive();
  test_parse_keywords_positive();
  test_parse_negative_becomes_default();
}
