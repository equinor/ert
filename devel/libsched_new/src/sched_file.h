#ifndef __SCHED_FILE_H__
#define __SCHED_FILE_H__

#include <stdio.h>
#include <time.h>
#include <sched_kw.h>

typedef struct sched_file_struct sched_file_type;

sched_file_type * sched_file_alloc();
void sched_file_free(sched_file_type *);
void sched_file_parse(sched_file_type *, time_t, const char *);

void sched_file_fprintf_i(const sched_file_type *, int, const char *);

void sched_file_fwrite(const sched_file_type *, FILE * stream);
sched_file_type * sched_file_fread_alloc(FILE * stream); 

int sched_file_get_nr_restart_files(const sched_file_type *);
int sched_file_iget_block_size(const sched_file_type *, int);
int sched_file_time_t_to_restart_file(const sched_file_type *, time_t);
time_t sched_file_iget_block_start_time(const sched_file_type *, int);
time_t sched_file_iget_block_end_time(const sched_file_type *, int);
sched_kw_type * sched_file_ijget_block_kw_ref(const sched_file_type *, int, int);

#endif
