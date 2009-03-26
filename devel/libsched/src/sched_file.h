#ifndef __SCHED_FILE_H__
#define __SCHED_FILE_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <time.h>
#include <sched_kw.h>

typedef struct sched_file_struct sched_file_type;


  typedef void * (sched_file_callback_ftype)( void * sched_kw , int restart_nr , void * arg);


sched_file_type * sched_file_alloc(time_t);
sched_file_type * sched_file_safe_cast(void * );
void 		  sched_file_free(sched_file_type *);
void 		  sched_file_parse(sched_file_type *, time_t, const char *);
void              sched_file_parse_append(sched_file_type *  , const char * );
sched_file_type * sched_file_parse_alloc(const char * , time_t);
void              sched_file_fprintf_i(const sched_file_type *, int, const char *);
void              sched_file_fprintf(const sched_file_type * sched_file, const char * file);
void              sched_file_fwrite(const sched_file_type *, FILE * stream);
sched_file_type * sched_file_fread_alloc(FILE * stream); 

int               sched_file_get_num_restart_files(const sched_file_type *);
int               sched_file_get_restart_nr_from_time_t(const sched_file_type *, time_t);
int               sched_file_get_restart_nr_from_days(const sched_file_type *  , double );
int               sched_file_iget_block_size(const sched_file_type *, int);
int               sched_file_time_t_to_restart_file(const sched_file_type *, time_t);

time_t            sched_file_iget_block_start_time(const sched_file_type *, int);
time_t            sched_file_iget_block_end_time(const sched_file_type *, int);
double            sched_file_iget_block_start_days(const sched_file_type * sched_file, int i);
double            sched_file_iget_block_end_days(const sched_file_type * sched_file, int i);
double            sched_file_get_sim_days(const sched_file_type * sched_file , int report_step);

sched_kw_type *   sched_file_ijget_block_kw_ref(const sched_file_type *, int, int);
int               sched_file_get_restart_file_from_time_t(const sched_file_type * , time_t );
void              sched_file_summarize(const sched_file_type *  , FILE * );
sched_file_type * sched_file_alloc_copy(const sched_file_type * , bool);


void 		  sched_file_update_blocks(sched_file_type * sched_file, int restart1 , int restart2 , sched_type_enum kw_type, sched_file_callback_ftype * callback, void * arg);
void 		  sched_file_update(sched_file_type * sched_file, sched_type_enum kw_type, sched_file_callback_ftype * callback, void * arg);


#ifdef __cplusplus
}
#endif
#endif
