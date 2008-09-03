#ifndef __SCHED_FILE_H__
#define __SCHED_FILE_H__
#include <stdio.h>
#include <time.h>

typedef struct sched_file_struct sched_file_type;

sched_file_type * sched_file_alloc();
void sched_file_free(sched_file_type *);
void sched_file_parse(sched_file_type *, time_t, const char *);

int sched_file_get_nr_report_steps(const sched_file_type *);
void sched_file_fprintf(const sched_file_type *, int, const char *);

void sched_file_fwrite(const sched_file_type *, FILE * stream);
sched_file_type * sched_file_fread_alloc(FILE * stream); 

#endif
