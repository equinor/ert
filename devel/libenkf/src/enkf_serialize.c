#include <stdbool.h>
#include <stdlib.h>
#include <enkf_serialize.h>
#include <enkf_types.h>
#include <util.h>




/**
   This is a support struct designed to hold information about a
   partly completed serialization. Observe that there are two reasons
   why the serialization of one object might be called repeatedly:

    * The serial vector is full - i.e. we run out of memory.  We are
    * using local analysis, and call several times to update different
      parts of the object.

*/

struct serial_state_struct {
  int        internal_offset;   /* How long into the nodes data we have completed the serialization. */
  int        serial_size;       /* The number of elements which have been serialized from this node - in this round. */
  size_t     offset;            /* The current offset into the serial vector. */
  bool       state_complete;    /* Whether we are done with serialization of this node. */
  state_enum state;
};



/*****************************************************************/

/*
  1. serialize: input : internal_offset
                output: elements_added , state_complete

  2. serial_state_update_forecast()

  3. EnkF update multiply X * serial_state.

  4. deserialize: input:  elements_added
                  output: updated internal_offste , state_complete

  5. serial_state_update_serialized()
*/





void serial_state_clear(serial_state_type * state) {
  state->internal_offset    = 0;
  state->state              = forecast;
  state->serial_size        = 0;
  state->state_complete     = false;
  state->offset             = 0;
}


serial_state_type * serial_state_alloc() {
  serial_state_type * state = util_malloc(sizeof * state , __func__);
  serial_state_clear(state);
  return state;
}


 void serial_state_free(serial_state_type * state) {
  free(state);
  state = NULL;
}



 bool serial_state_do_serialize(const serial_state_type * state) {
  if (state->state == forecast)
    return true;
  else
    return false;
}


 bool serial_state_do_deserialize(const serial_state_type * state) {
  if (state->state == serialized)
    return true;
  else
    return false;
}


 int serial_state_get_internal_offset(const serial_state_type * state) {
  return state->internal_offset;
}



 void serial_state_update_forecast(serial_state_type * state , size_t offset , int elements_added , bool complete) {
  state->serial_size    = elements_added;
  state->state_complete = complete;
  state->state          = serialized;
  state->offset         = offset;
}



 void serial_state_update_serialized(serial_state_type * state , int new_internal_offset) {
  if (state->state_complete) {
    state->state           = analyzed;
    state->serial_size     = -1;
    state->internal_offset = -1;
  } else {
    state->state           = forecast;
    state->serial_size     = -1;
    state->state_complete  = false;
    state->internal_offset = new_internal_offset;
  }
}


 void serial_state_init_deserialize(const serial_state_type * serial_state , int * internal_offset , size_t * serial_offset, int * node_serial_size) {
  *internal_offset  = serial_state->internal_offset;
  *serial_offset    = serial_state->offset;
  *node_serial_size = serial_state->serial_size;
}


/*****************************************************************/

size_t enkf_serialize(const void * __node_data, ecl_type_enum node_type ,  const bool * active , size_t node_offset , size_t node_size , double * serial_data , 
		      size_t serial_size , size_t serial_offset , int serial_stride ,  bool * complete) {
  
  size_t node_index;
  size_t serial_index = 0;

  if (node_type == ecl_double_type) {
    /* Serialize double -> double */
    const  double * node_data = (const double *) __node_data;
#include "serialize.h"
  } else if (node_type == ecl_float_type) {
    /* Serialize float -> double */
    const  float * node_data = (const float *) __node_data;
#include "serialize.h"
  } else 
    util_abort("%s: internal error: trying to serialize unserializable type:%s \n",__func__ , ecl_util_type_name( node_type ));

  return serial_index;

}


size_t enkf_deserialize(void * __node_data      	  , 
			ecl_type_enum node_type 	  , 
			const bool * active     	  , 
			size_t node_offset      	  , 
			size_t node_size        	  , 
			size_t node_serial_size 	  ,  
			const double * serial_data , 
			size_t serial_offset       , 
			int serial_stride) {
  
  size_t serial_index = 0;
  size_t node_index;
  size_t new_node_offset = 0;
  int    last_node_index = util_int_min(node_size , node_offset + node_serial_size);
  if (last_node_index < (node_size - 1))
    new_node_offset = last_node_index;
  else
    new_node_offset = 0;
  
  if (node_type == ecl_double_type) {
    double * node_data = (double *) __node_data;
#include "deserialize.h"
  } else if (node_type == ecl_float_type) {
    float * node_data = (float *) __node_data;
#include "deserialize.h"
  } else 
    util_abort("%s: internal error: trying to deserialize unserializable type:%s \n",__func__ , ecl_util_type_name( node_type ));

  return new_node_offset;
}

