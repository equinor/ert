#ifndef __ENKF_SERIALIZE_H__
#define __ENKF_SERIALIZE_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <stdbool.h>
#include <ecl_util.h>

typedef struct serial_state_struct  serial_state_type;
typedef struct serial_vector_struct serial_vector_type;


void                serial_state_clear(serial_state_type * );
serial_state_type * serial_state_alloc();
void 		    serial_state_free(serial_state_type * );
bool 		    serial_state_do_serialize(const serial_state_type * );
bool                serial_state_complete(const serial_state_type * );


void                 serial_vector_free(serial_vector_type * );
serial_vector_type * serial_vector_alloc( size_t , int);
int                  serial_vector_get_stride(const serial_vector_type * );
double *             serial_vector_get_data(const serial_vector_type * );

/*****************************************************************/

void   enkf_deserialize_part(void * , bool , int , int , int , ecl_type_enum , int , const int * , serial_state_type * , const serial_vector_type *);
size_t enkf_serialize_part  (const void * , bool , int , int , int , ecl_type_enum ,  int , const int * , serial_state_type * , size_t , serial_vector_type *);


void   enkf_deserialize(void * , int , ecl_type_enum , int , const int * , serial_state_type * , const serial_vector_type *);
size_t enkf_serialize  (const void * , int , ecl_type_enum ,  int , const int * , serial_state_type * , size_t , serial_vector_type *);


#ifdef __cplusplus
}
#endif
#endif
