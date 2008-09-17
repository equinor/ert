#include <stdbool.h>
#include <stdlib.h>
#include <enkf_serialize.h>
#include <enkf_types.h>
#include <util.h>


/** This is heavy shit ... */


/**
   This file handles serialization and deserialization of the
   enkf_nodes. This is at the very core of the EnKF update
   algorithm. The final update step is written:

       A' = XA

   i.e. it is linear algeabra, and we need(?) to write the various
   objects in the form of an ensemble matrix, this is the process we
   call serialization. Then the linear algebra update is performed,
   and afterwards we must read the data back from the serial vector to
   the enkf_node object, this is deserialization.


                                         
   
  ===============                                    ===============
  | PORO-1      |                		     | PORO-2      |
  |--------------                    Member 2        |--------------   
  |             |                    ========        |             |       
  | P0 ... P10  |        Member 1       |	     | P0 ... P10  |
  ===============        ========       |	     ===============
       /|\                   |      -----                 /|\ 
        |                    |      |                      |
        |                   \|/    \|/                     |
        |                  -------------                   | 
        |                  | P0     P0 |                   |
        |                  | P1     P1 |                   |
        ------------------>| P2     P2 |<-------------------
                           | P3     P3 |
                           | P4     P4 |    
                           | ··········|             ==============     
                           | R1     R1 |	     | RELPERM-2  |
  ==============    ------>| R2     R2 |<----------->|------------|
  | RELPERM-1  |    |      | R3     R3 |	     |            |
  |------------|<----      | ··········|    	     | R0 ... R5  |
  |            |	   | F2	    F2 |	     ==============
  | R0 ... R5  |      ---->| F3	    F3 |
  ==============      |    | F4	    F4 |<-----
                      |    | F6	    F6 |     |
                      |    -------------     |
                      |                      |           ==============
                      | 	 	     |           | FAULT-1    |
                      |                      ----------->|------------|
  ==============      |				         |            |
  | FAULT-1    |      |				         | F0 ... F6  |
  |------------|<------				         ==============
  |            |
  | F0 ... F6  |
  ==============



This figure shows the following:

 1. Three different nodes called PORO, RELPERM and FAULT
    respectively. The PORO node consists of eleven elements (P0
    ... P10), whereas the RELPERM and FAULT nodes contain six and
    seven elements.

 2. The ensemble consists of two members (i.e. there is PORO-1 and
    PORO-2.).

 3. The members have been serialized into a a large vector where
    everything comes ordered. Observe that *NOT* all elements from the
    members have been inserted into the large vector, i.e. for the
    PORO fields we only have elements P0 .. P4; this is because (for
    some reason) not all elements were active.


Each of the enkf_node functions have their own xxx_serialize and
xxx_deserialize functions, however these functions SHOULD call the
enkf_serialize() and enkf_deserialize() functions in this
file. (Rolling your own serialize / deserialize functions at the
lowest level is a SERIOUS CRIME.)

The illustration above shows three different enkf_node objects which
have been COMPLETELY serialized. One of the reasons the code is so
complex is that it is supposed to handle situations where the serial
vector is to small to hold everything, and repeated calls to serialize
& deserialize must be performed to complete the thing.
*/


/*****************************************************************/

/**
   This is a support struct designed to hold information about a
   partly completed serialization. Observe that there are two reasons
   why the serialization of one object might be called repeatedly:

    * The serial vector is full - i.e. we run out of memory.  We are
    * using local analysis, and call several times to update different
      parts of the object.

*/

struct serial_state_struct {
  int        node_index1;       /* The first node_index which is currently serialized. */
  int        node_index2;       /* The last node_index which is (not) currently serialized. */
  int        serial_node_size;  /* The number of elements which have been serialized from this node - in this round of serialization. */
  size_t     serial_offset;     /* The current offset into the serial vector. */
  bool       complete;          /* Whether we are done with serialization of this node. */
  state_enum state;
};



/*****************************************************************/

void serial_state_clear(serial_state_type * state) {
  state->node_index1        = 0;
  state->node_index2        = 0;
  state->serial_node_size   = 0;
  state->serial_offset      = 0;
  
  state->complete           = false;
  state->state              = forecast;
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




bool serial_state_complete(const serial_state_type * state) { return state->complete; }


/*****************************************************************/

/** 
    Observe: The first five arguments regard the node which is
    currently being serialized, whereas the last four are about the
    serial vector which will hold the serialized data.
*/

size_t enkf_serialize(const void * __node_data	       ,   /* The data of the input node - pointer to either float or double */
		      int           node_size          ,   /* The size (number of elements) of __node_data array. */
		      ecl_type_enum node_type 	       ,   /* The underlying data type of __node_data. */
		      const bool * active     	       ,   /* An active flag for the data - can be NULL if everything is active. */      
		      serial_state_type * serial_state ,   /* Holding the state of the current serialization of this node. */
		      /*-- Above: node data --- Below: serial data --*/
		      double * serial_data ,               /* The large vector which accepts the data. */
		      size_t serial_size   ,               /* The size of the serial vector. */
		      size_t serial_offset ,               /* The offset in the serial vector we are starting on. */
		      int serial_stride) {                 /* The stride in the serial vector.*/

  int    node_index1    = serial_state->node_index1;
  size_t elements_added = 0;
  
  if (!serial_state->complete) {
    /* If the previous _exactly_ used all memory we will have serial_size == serial_offset - and should return immediatebly. */
    if (serial_offset < serial_size) {  
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
      
      
      /* 
	 This is is "all" the information we are going to need in the
	 subsequent deserialization.
      */
      serial_state->state            = serialized;
      serial_state->serial_node_size = elements_added;
      serial_state->serial_offset    = serial_offset;
      /* 
	 serial_state->node_index2 and serial_state->complete are set in serialize.h.
      */
    }
  }
  return elements_added;
}





void enkf_deserialize(void * __node_data      	  	, /* The data of the node which will accept updated data. */
		      int      node_size          	, /* The total size of the node we are updating. */
		      ecl_type_enum node_type 	  	, /* The underlying type (double || float) of the node's storage. */
		      const bool * active     	  	, /* Active / inactive flag - can be NULL for all elements active. */
		      serial_state_type * serial_state  , /* Holding the state of the current serialization of this node. */
		      /*-- Above: node data ---  Below: serial data --*/
		      const double * serial_data        , /* The serial vector which we will now be reading out of. */     
		      int serial_stride) {                /* The stride in the serial vector.*/

  if (serial_state->state == serialized) {
    size_t serial_offset = serial_state->serial_offset;
    int    node_index1   = serial_state->node_index1;
    int    node_index2   = serial_state->node_index2;
    
    
    if (node_type == ecl_double_type) {
      double * node_data = (double *) __node_data;
#include "deserialize.h"
    } else if (node_type == ecl_float_type) {
      float * node_data = (float *) __node_data;
#include "deserialize.h"
    } else 
      util_abort("%s: internal error: trying to deserialize unserializable type:%s \n",__func__ , ecl_util_type_name( node_type ));
    
    if (serial_state->complete) {
      serial_state->state            = analyzed;
      serial_state->serial_node_size = -1;
    } else 
      serial_state->node_index1 = serial_state->node_index2;  /* This is where the next serialization should start. */
  }
}

