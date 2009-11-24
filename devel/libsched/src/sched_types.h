#ifndef __SCHED_TYPES_H__
#define __SCHED_TYPES_H__
#ifdef __cplusplus 
extern "C" {
#endif

/**
   Contains numerous typedefs for the types used in the sched_kw keywords. 
*/




typedef enum { OPEN  = 1, 
               STOP  = 2, 
               SHUT  = 3, 
               AUTO  = 4 }      wconinje_status_enum;
#define STATUS_OPEN_STRING "OPEN"
#define STATUS_STOP_STRING "STOP"
#define STATUS_SHUT_STRING "SHUT"
#define STATUS_AUTO_STRING "AUTO"

  
/**
   There is no default injector type.
*/
typedef enum {WATER , GAS  , OIL} sched_phase_type;

  
/**
   There is no default cmode value. 
*/
typedef enum {RATE  , RESV , BHP , THP , GRUP} wconinje_control_enum;
#define CONTROL_RATE_STRING  "RATE"
#define CONTROL_RESV_STRING  "RESV"
#define CONTROL_BHP_STRING   "BHP"
#define CONTROL_THP_STRING   "THP"
#define CONTROL_GRUP_STRING  "GRUP"



/*****************************************************************/



typedef enum {WCONHIST =  0, 
              DATES    =  1, 
              COMPDAT  =  2, 
              TSTEP    =  3, 
              TIME     =  4,   /* Not implemented support */
              WELSPECS =  5, 
              GRUPTREE =  6,
              INCLUDE  =  7,   
              UNTYPED  =  8,
              WCONINJ  =  9,
              WCONINJE = 10, 
              WCONINJH = 11,
              WCONPROD = 12,
              NUM_SCHED_KW_TYPES = 13} sched_kw_type_enum;
              

sched_kw_type_enum sched_kw_type_from_string(const char * kw_name);
const char *       sched_kw_type_name(sched_kw_type_enum kw_type);

/*****************************************************************/  

sched_phase_type sched_phase_type_from_string(const char * type_string);
const char *     sched_phase_type_string(sched_phase_type type);

#ifdef __cplusplus 
}
#endif
#endif
