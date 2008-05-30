#ifndef __BASIC_QUEUE_DRIVER_H__
#define __BASIC_QUEUE_DRIVER_H__

typedef enum {ecl_queue_null          = 0 ,   /* For a queue node which has been allocated - but not "added" with a ecl_queue_add_job() call. */
	      ecl_queue_waiting       = 1 ,   /* A node which is waiting in the internal queue. */
	      ecl_queue_pending       = 2 ,   /* A node which is pending - a status returned by the external system. I.e LSF */
	      ecl_queue_running       = 3 ,   /* The job is running */
	      ecl_queue_done          = 4 ,   /* The job is done - but we have not yet checked if the target file is produced */
	      ecl_queue_exit          = 5 ,   /* The job has exited - check attempts to determine if we retry or go to complete_fail */
	      ecl_queue_complete_OK   = 6 ,   
	      ecl_queue_complete_FAIL = 7 ,
	      ecl_queue_restart       = 8 ,
	      ecl_queue_max_state     = 9 } ecl_job_status_type;


typedef struct basic_queue_driver_struct basic_queue_driver_type;
typedef struct basic_queue_job_struct    basic_queue_job_type;

typedef basic_queue_job_type * (submit_job_ftype)  	 (basic_queue_driver_type * , int , const char * , const char * , const char * , const char* , const char * , const char * , const char *);
typedef void                   (abort_job_ftype)   	 (basic_queue_driver_type * , basic_queue_job_type * );
typedef ecl_job_status_type    (get_status_ftype)  	 (basic_queue_driver_type * , basic_queue_job_type * );
typedef void                   (free_job_ftype)    	 (basic_queue_driver_type * , basic_queue_job_type * );
typedef void                   (free_queue_driver_ftype) (basic_queue_driver_type *);

void basic_queue_driver_assert_cast(const basic_queue_driver_type * );
void basic_queue_driver_init(basic_queue_driver_type * );
void basic_queue_job_assert_cast(const basic_queue_job_type * );
void basic_queue_job_init(basic_queue_job_type * );


struct basic_queue_job_struct {
  int __id;
};

#define BASIC_QUEUE_DRIVER_FIELDS        \
submit_job_ftype  	* submit;        \
free_job_ftype    	* free_job;      \
abort_job_ftype   	* abort_f;       \
get_status_ftype  	* get_status;    \
free_queue_driver_ftype * free_driver;   \
int __id;        


struct basic_queue_driver_struct {
  BASIC_QUEUE_DRIVER_FIELDS
};




#endif
