#ifndef ERT_TIME_MAP_H
#define ERT_TIME_MAP_H

#include <time.h>

#include <ert/ecl/ecl_sum.h>
#include <ert/tooling.hpp>
#include <ert/util/int_vector.h>
#include <ert/util/type_macros.h>

typedef struct time_map_struct time_map_type;

UTIL_SAFE_CAST_HEADER(time_map);
UTIL_IS_INSTANCE_HEADER(time_map);

bool time_map_try_summary_update(time_map_type *map,
                                 const ecl_sum_type *ecl_sum);
extern "C" bool time_map_try_update(time_map_type *map, int step, time_t time);
extern "C" bool time_map_attach_refcase(time_map_type *time_map,
                                        const ecl_sum_type *refcase);
bool time_map_has_refcase(const time_map_type *time_map);
void time_map_clear(time_map_type *map);
bool time_map_equal(const time_map_type *map1, const time_map_type *map2);
extern "C" time_map_type *time_map_alloc();
extern "C" void time_map_free(time_map_type *map);
bool time_map_update(time_map_type *map, int step, time_t time);
std::string time_map_summary_update(time_map_type *map,
                                    const ecl_sum_type *ecl_sum);
extern "C" time_t time_map_iget(time_map_type *map, int step);
extern "C" void time_map_fwrite(time_map_type *map, const char *filename);
extern "C" void time_map_fread(time_map_type *map, const char *filename);
extern "C" bool time_map_fscanf(time_map_type *map, const char *filename);
extern "C" double time_map_iget_sim_days(time_map_type *map, int step);
extern "C" int time_map_get_last_step(time_map_type *map);
extern "C" int time_map_get_size(time_map_type *map);
time_t time_map_get_start_time(time_map_type *map);
time_t time_map_get_end_time(time_map_type *map);
double time_map_get_end_days(time_map_type *map);
bool time_map_is_readonly(const time_map_type *tm);
int_vector_type *time_map_alloc_index_map(time_map_type *map,
                                          const ecl_sum_type *ecl_sum);
extern "C" int time_map_lookup_time(time_map_type *map, time_t time);
extern "C" int time_map_lookup_days(time_map_type *map, double sim_days);
extern "C" int time_map_lookup_time_with_tolerance(time_map_type *map,
                                                   time_t time,
                                                   int seconds_before_tolerance,
                                                   int seconds_after_tolerance);

#endif
