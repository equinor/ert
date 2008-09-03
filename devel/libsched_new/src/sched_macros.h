#ifndef __SCHED_MACROS_H___
#define __SCHED_MACROS_H___



/***********************************
Macros for "voidifications" of the data handlers.

  - *_IMPL     - The implementation
  - *_HEADER   - Header
  - None.      - Name.

***********************************/

#define KW_FSCANF_ALLOC(KW) __sched_kw_## KW ##_fscanf_alloc
#define KW_FWRITE(KW)       __sched_kw_## KW ##_fwrite
#define KW_FREAD_ALLOC(KW)  __sched_kw_## KW ##_fread_alloc
#define KW_FREE(KW)         __sched_kw_## KW ##_free
#define KW_FPRINTF(KW)      __sched_kw_## KW ##_fprintf

#define GET_DATA_HANDLERS(DH, KWNAME) \
DH.fscanf_alloc = KW_FSCANF_ALLOC(KWNAME) ; \
DH.free         = KW_FREE(        KWNAME) ; \
DH.fprintf      = KW_FPRINTF(     KWNAME) ; \
DH.fwrite       = KW_FWRITE(      KWNAME) ; \
DH.fread_alloc  = KW_FREAD_ALLOC( KWNAME)   \

/*******************************************************************/

#define KW_FSCANF_ALLOC_IMPL(KW) \
void * __sched_kw_## KW ##_fscanf_alloc(FILE * stream, bool *at_eof, const char * kw_name) \
{                                                                                          \
  return (void *) sched_kw_## KW ##_fscanf_alloc(stream, at_eof, kw_name);                 \
}                                                                                          \

#define KW_FWRITE_IMPL(KW)                                               \
void   __sched_kw_## KW ##_fwrite(const void * kw , FILE * stream)       \
{                                                                        \
  sched_kw_## KW ##_fwrite((const sched_kw_## KW ##_type *) kw, stream); \
}                                                                        \

#define KW_FREAD_ALLOC_IMPL(KW)                             \
void * __sched_kw_ ## KW ## _fread_alloc(FILE * stream)     \
{                                                           \
  return (void *) sched_kw_ ## KW ## _fread_alloc(stream);  \
}                                                           \

#define KW_FPRINTF_IMPL(KW)                                                \
void   __sched_kw_## KW ##_fprintf(const void * kw, FILE * stream)         \
{                                                                          \
  sched_kw_## KW ##_fprintf((const sched_kw_## KW ##_type *) kw, stream);  \
}                                                                          \

#define KW_FREE_IMPL(KW)                                \
void   __sched_kw_## KW ##_free(void * kw)              \
{                                                       \
  sched_kw_## KW ##_free((sched_kw_## KW ##_type *) kw);\
}                                                       \

/*******************************************************************/

#define KW_FSCANF_ALLOC_HEADER(KW)                                      \
void * __sched_kw_## KW ##_fscanf_alloc(FILE * , bool *, const char *); \

#define KW_FWRITE_HEADER(KW)                                    \
void   __sched_kw_## KW ##_fwrite(const void * , FILE *);       \

#define KW_FREAD_ALLOC_HEADER(KW)                      \
void * __sched_kw_ ## KW ## _fread_alloc(FILE * );     \

#define KW_FPRINTF_HEADER(KW)                              \
void   __sched_kw_## KW ##_fprintf(const void *, FILE * ); \

#define KW_FREE_HEADER(KW)               \
void   __sched_kw_## KW ##_free(void *); \

#endif
