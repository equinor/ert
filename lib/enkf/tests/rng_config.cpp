/*
   Copyright (C) 2017  Equinor ASA, Norway.

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

#include <ert/util/test_util.h>
#include <ert/util/test_work_area.h>

#include <ert/enkf/rng_config.hpp>
#include <ert/res_util/res_log.hpp>



#define MAX_INT 999999

static void create_config(
        const char * user_config_file,
        const char * random_seed)
{
  FILE * stream = util_fopen(user_config_file, "w");
  fprintf(stream, "NUM_REALIZATIONS 17\n");
  if(random_seed)
    fprintf(stream, "RANDOM_SEED %s\n", random_seed);
  fclose(stream);
}

static char * alloc_read_random_seed(const char * log_file)
{
  FILE * stream = util_fopen(log_file, "r");
  char word [256];
  char random_seed [256];
  while(fscanf(stream, "%s", word) == 1)
    if (strcmp("RANDOM_SEED", word) == 0)
      fscanf(stream, "%s", random_seed);

  fclose(stream);

  return util_alloc_string_copy(random_seed);
}

void test_init()
{
  test_work_area_type * work_area = test_work_area_alloc("rng_config");
  res_log_init_log(LOG_DEBUG, "log", true);

  const char * config_file = "my_rng_config";
  const char * random_seed = "13371338";

  create_config(config_file, random_seed);

  rng_config_type * rng_config = rng_config_alloc_load_user_config(config_file);
  test_assert_string_equal(random_seed, rng_config_get_random_seed(rng_config));

  rng_manager_free(rng_config_alloc_rng_manager(rng_config));

  // To get the random seed written to the log

  char * logged_random_seed = alloc_read_random_seed("log");
  test_assert_true(strlen(logged_random_seed) > 0);

  free(logged_random_seed);
  free(rng_config);
  free(work_area);
}

static void alloc_reproduced_rng_config(
        const char * random_seed,
        rng_config_type ** orig_rng_config,
        rng_config_type ** rep_rng_config,
        rng_manager_type ** orig_rng_man,
        rng_manager_type ** rep_rng_man)
{
  test_work_area_type * work_area = test_work_area_alloc("rng_config");
  res_log_init_log(LOG_DEBUG, "log", true);

  const char * config_file = "my_rng_config";
  create_config(config_file, random_seed);
  *orig_rng_config = rng_config_alloc_load_user_config(config_file);

  rng_manager_type * rng_man = rng_config_alloc_rng_manager(*orig_rng_config);
  if(orig_rng_man)
    *orig_rng_man = rng_man;
  else
    rng_manager_free(rng_man);

  const char * rep_config_file = "rep_config";
  char * logged_random_seed = alloc_read_random_seed("log");
  create_config(rep_config_file, logged_random_seed);
  *rep_rng_config = rng_config_alloc_load_user_config(rep_config_file);

  if(rep_rng_man)
    *rep_rng_man = rng_config_alloc_rng_manager(*rep_rng_config);

  free(logged_random_seed);
  free(work_area);
}

void test_reproducibility(const char * random_seed)
{

  rng_config_type * orig_rng_config;
  rng_config_type * rep_rng_config;

  rng_manager_type * orig_rng_man;
  rng_manager_type * rep_rng_man;

  alloc_reproduced_rng_config(random_seed,
                              &orig_rng_config, &rep_rng_config,
                              &orig_rng_man, &rep_rng_man);

  test_assert_not_NULL(orig_rng_man);
  test_assert_not_NULL(rep_rng_man);

  rng_type * orig_rng_0   = rng_manager_iget(orig_rng_man, 0);
  rng_type * orig_rng_100 = rng_manager_iget(orig_rng_man, 100);

  rng_type * rep_rng_0   = rng_manager_iget(rep_rng_man, 0);
  rng_type * rep_rng_100 = rng_manager_iget(rep_rng_man, 100);

  test_assert_int_equal(rng_get_int(orig_rng_0, MAX_INT), rng_get_int(rep_rng_0, MAX_INT));
  test_assert_int_equal(rng_get_int(orig_rng_100, MAX_INT), rng_get_int(rep_rng_100, MAX_INT));
}

int main(int argc , char ** argv) {
  test_init();
  test_reproducibility(NULL); // Random seed
  test_reproducibility("42");
  test_reproducibility("423543854372895743289507289532");
  test_reproducibility("423543854372895743289507289532423543854372895743289507289532");
}
