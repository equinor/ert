#include <stdlib.h>
#include <util.h>
#include <ecl_static_kw.h>
#include <ecl_kw.h>
#include <enkf_macros.h>



struct ecl_static_kw_struct {
  int           __type_id;
  ecl_kw_type * ecl_kw;
  
  /*-----------------------------------------------------------------*/
  /* The fields below here are a fuxxxg hack to support multiple
     keywords with the same 'name' - see documentation below. */
  
  int  __kw_count;      /* Which mumber in the series this instance is - starting at 0.*/
  bool __write_mode;    /* Whether we are currently writing (the static keyword to the enkf_fs 'database'). */
  int  __report_step;   /* The currently active report_step .*/
};


/**
   It is an assumption quite heavily immersed in the enkf code that
   the enkf_state objects should have unique keys identifying the
   various nodes. Now, unfortunately it turns out that the keywords
   relating to AQUIFER properties (at least) can appear several times
   in a restart file, and we must handle that.
   
   The implementation is quite naive - observe the following points:

    o It has been a design principle that the calling scope should
      _not_ have to do any book keeping on the static ecl_kw instances.
      
    o The ecl_static_kw objects can only store _one_ ecl_kw instance
      at a time. 

   The are several responsibilities on the calling scope (which are in
   accordance with the current implementation):

    o Calling scope must call ecl_static_kw_inc_counter() _before_ the
      calling enkf_fs with the node. It must call with write_mode ==
      true when the static keywords are stored, and with write_mode ==
      false when the static keywords are loaded again.

    o Calling scope must make sure that the ecl_kw instance is freed
      immediately after use by calling enkf_node_free_data().

   The implementation basicilly works by increasing an integer
   counter, which is reset to zero everytime write_mode _or_
   report_step changes value.
*/




/**
   The input state is defined by the parameters write_mode and
   report_step. If they differ from the currently stored state the
   counter is reset to 0, otherwise it is increased.
*/

void ecl_static_kw_inc_counter(ecl_static_kw_type * ecl_static, bool write_mode , int report_step) {
  
  if (ecl_static->__write_mode != write_mode)             /* Changing reading <-> writing */
    ecl_static->__kw_count = 0;
  else if (ecl_static->__report_step != report_step)      /* Changing report_step */
    ecl_static->__kw_count = 0;
  else
    ecl_static->__kw_count++;                             /* Increase counter */
  
  ecl_static->__write_mode  = write_mode;
  ecl_static->__report_step = report_step;
}


int ecl_static_kw_get_report_step(const ecl_static_kw_type * ecl_static) {
  return ecl_static->__report_step;
}



/**
   Used by the filessystem function reading/writing spesific instances to disk.
*/
int ecl_static_kw_get_counter(const ecl_static_kw_type * ecl_static) {
  if (ecl_static->__kw_count < 0)
    util_abort("%s: internal error: __kw_count = %d \n",__func__ , ecl_static->__kw_count);
  
  return ecl_static->__kw_count;
}





/**


*/

ecl_static_kw_type * ecl_static_kw_alloc( ) {  
  ecl_static_kw_type * static_kw = util_malloc(sizeof *static_kw , __func__);
  static_kw->__report_step = -1;
  static_kw->__write_mode  = false;
  static_kw->ecl_kw    	   = NULL;
  static_kw->__type_id 	   = STATIC;
  return static_kw;
}

/*
  ptr is pure dummy to satisfy the api.
*/
void * ecl_static_kw_alloc__(const void *ptr) {  
  return ecl_static_kw_alloc( );
}


void ecl_static_kw_ecl_write(const ecl_static_kw_type * ecl_static, const char * run_path /* Not used*/  , const char * path /* Not used */, fortio_type * fortio) {
  ecl_kw_fwrite(ecl_static->ecl_kw , fortio);
}


ecl_kw_type * ecl_static_kw_ecl_kw_ptr(const ecl_static_kw_type * ecl_static) { return ecl_static->ecl_kw; }


ecl_static_kw_type * ecl_static_kw_copyc(const ecl_static_kw_type *src) {
  ecl_static_kw_type * new = ecl_static_kw_alloc();
  if (src->ecl_kw != NULL)
    new->ecl_kw = ecl_kw_alloc_copy(src->ecl_kw);
  return new;
}


void ecl_static_kw_free_data(ecl_static_kw_type * kw) {
  if (kw->ecl_kw != NULL) ecl_kw_free(kw->ecl_kw);
  kw->ecl_kw = NULL;
}


void ecl_static_kw_free(ecl_static_kw_type * kw) {
  ecl_static_kw_free_data(kw);
  free(kw);
}


void ecl_static_kw_init(ecl_static_kw_type * ecl_static_kw, const ecl_kw_type * ecl_kw) {
  if (ecl_static_kw->ecl_kw != NULL)
    util_abort("%s: internal error: trying to assign ecl_kw to ecl_static_kw which is already set.\n",__func__);
  
  ecl_static_kw->ecl_kw = ecl_kw_alloc_copy(ecl_kw);
}


void ecl_static_kw_fread(ecl_static_kw_type * ecl_static_kw , FILE * stream) {
  enkf_util_fread_assert_target_type(stream , STATIC);
  if (ecl_static_kw->ecl_kw != NULL)
    util_abort("%s: internal error: trying to assign ecl_kw to ecl_static_kw which is already set.\n",__func__);
  ecl_static_kw->ecl_kw = ecl_kw_fread_alloc_compressed(stream);
}


bool ecl_static_kw_fwrite(const ecl_static_kw_type * ecl_static_kw , FILE * stream) {
  enkf_util_fwrite_target_type(stream , STATIC);
  ecl_kw_fwrite_compressed(ecl_static_kw->ecl_kw , stream);
  return true;
}



/**
This is a pure dummy, memory is handled differently for this object
type:

  o The ecl_kw_type instance holding the data is boostrapped on fread().
  
  o The whole ecl_kw_type instances is free'd on free_data.

*/

void ecl_static_kw_realloc_data(ecl_static_kw_type * ecl_static_kw) {
  return ;
}




/*****************************************************************/
SAFE_CAST(ecl_static_kw , STATIC)
VOID_FREE(ecl_static_kw)
VOID_FREE_DATA(ecl_static_kw)
VOID_FWRITE (ecl_static_kw)
VOID_ECL_WRITE (ecl_static_kw)
VOID_FREAD  (ecl_static_kw)
VOID_COPYC(ecl_static_kw)
VOID_REALLOC_DATA(ecl_static_kw)
