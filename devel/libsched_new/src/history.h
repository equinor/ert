#ifndef __HISTORY_H__
#define __HISTORY_H__


#include <stdbool.h>
#include <stdio.h>
#include <ecl_sum.h>
#include <sched_file.h>



typedef struct history_struct history_type;


// Manipulators.
void           history_free(history_type *);
void           history_fwrite(const history_type *, FILE * stream);
history_type * history_fread_alloc(FILE * stream);
history_type * history_alloc_from_sched_file(const sched_file_type *);
void           history_realloc_from_summary(history_type *, const ecl_sum_type *);



// Accessors.
int    history_get_num_restarts(const history_type *);
double history_get_well_var(const history_type * , int, const char *, const char *, bool *);
double history_get_group_var(const history_type *, int, const char *, const char *, bool *);
#endif
