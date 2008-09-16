#ifndef __SERIAL_STATE_H__
#define __SERIAL_STATE_H__
#include <stdbool.h>

typedef struct serial_state_struct serial_state_type;


void                serial_state_clear(serial_state_type * );
serial_state_type * serial_state_alloc();
void 		    serial_state_free(serial_state_type * );
bool 		    serial_state_do_serialize(const serial_state_type * );
bool 		    serial_state_do_deserialize(const serial_state_type * );
int                 serial_state_get_internal_offset(const serial_state_type * );
void 		    serial_state_update_forecast(serial_state_type * , size_t , int , bool );
void 		    serial_state_update_serialized(serial_state_type * , int );
void 		    serial_state_init_deserialize(const serial_state_type * , int * , size_t * , int * );


#endif
