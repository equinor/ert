#ifndef __BASIC_QUEUE_DRIVER_H__
#define __BASIC_QUEUE_DRIVER_H__

typedef enum {ecl_queue_null = 0, ecl_queue_waiting, ecl_queue_pending , ecl_queue_running , ecl_queue_done , ecl_queue_exit} ecl_job_status_type;

typedef struct basic_queue_driver_struct basic_queue_driver_type;
typedef struct basic_queue_job_struct    basic_queue_job_type;

typedef basic_queue_job_type * (submit_job_ftype)  (basic_queue_driver_type * , int , const char* , const char * , const char * , const char *);
typedef void                   (clean_job_ftype)   (basic_queue_driver_type * , basic_queue_job_type * );
typedef void                   (abort_job_ftype)   (basic_queue_driver_type * , basic_queue_job_type * );
typedef ecl_job_status_type    (get_status_ftype)  (basic_queue_driver_type * , basic_queue_job_type * );



void basic_queue_driver_assert_cast(const basic_queue_driver_type * );
void basic_queue_driver_init(basic_queue_driver_type * );
void basic_queue_job_assert_cast(const basic_queue_job_type * );
void basic_queue_job_init(basic_queue_job_type * );


struct basic_queue_job_struct {
  int __id;
};


struct basic_queue_driver_struct {
  int __id;
  submit_job_ftype * submit;
  clean_job_ftype  * clean;
  abort_job_ftype  * abort_f;
  get_status_ftype * get_status;
};




#endif
