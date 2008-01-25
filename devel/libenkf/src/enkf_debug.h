#ifdef  DEBUG
#define DEBUG_DECLARE   enkf_impl_type __impl_type;
#define DEBUG_ASSIGN(v) v->__impl_type = TARGET_TYPE;
#define DEBUG_ASSERT(v) if (v->__impl_type != TARGET_TYPE) { fprintf(stderr,"%s: Arrived with  __impl_type = %d - should be: %d - aborting. \n",__func__ , v->__impl_type , TARGET_TYPE); abort(); }
#else
#define DEBUG_DECLARE
#define DEBUG_ASSIGN(v)
#define DEBUG_ASSERT(v)
#endif
