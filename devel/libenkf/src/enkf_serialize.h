#ifndef __ENKF_SERIALIZE_H__
#define __ENKF_SERIALIZE_H__
#include <stdlib.h>
#include <stdbool.h>
#include <ecl_util.h>

typedef struct serial_state_struct serial_state_type;


void                serial_state_clear(serial_state_type * );
serial_state_type * serial_state_alloc();
void 		    serial_state_free(serial_state_type * );
bool 		    serial_state_do_serialize(const serial_state_type * );

/*****************************************************************/

size_t enkf_serialize  (const void * , int , ecl_type_enum ,  const bool * , serial_state_type * , double * , size_t , size_t , int);
void   enkf_deserialize(void * , int , ecl_type_enum , const bool * , serial_state_type * , const double * , int );


#endif
