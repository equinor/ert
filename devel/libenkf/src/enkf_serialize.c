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
  | PORO-1      |                                    | PORO-2      |
  |--------------                    Member 2        |--------------   
  |             |                    ========        |             |       
  | P0 ... P10  |        Member 1       |            | P0 ... P10  |
  ===============        ========       |            ===============
       /|\                   |      -----                 /|\ 
        |                    |      |                      |
        |                   \|/    \|/                     |
        |                                                  | 
        |                  [ P0     P0 ]                   |
        |                  [ P1     P1 ]                   |
        ------------------>[ P2     P2 ]<-------------------
                           [ P3     P3 ]
                           [ P4     P4 ]    
                           [ ··········]             ==============     
                           [ R1     R1 ]             | RELPERM-2  |
  ==============    ------>[ R2     R2 ]<----------->|------------|
  | RELPERM-1  |    |      [ R3     R3 ]             |            |
  |------------|<----      [ ··········]             | R0 ... R5  |
  |            |           [ F2     F2 ]             ==============
  | R0 ... R5  |      ---->[ F3     F3 ]
  ==============      |    [ F4     F4 ]<-----
                      |    [ F6     F6 ]     |
                      |                      |
                      |                      |           ==============
                      |                      |           | FAULT-1    |
                      |                      ----------->|------------|
  ==============      |                                  |            |
  | FAULT-1    |      |                                  | F0 ... F6  |
  |------------|<------                                  ==============
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


About stride
============
In the enkf update the ensemble matrix A is just that - a matrix,
however in this elegant high-level language it is of course
implemented as one long linear vector. The matrix is implemented such
that the 'member-direction' is fastest running index.  Consider the
following ensemble matrix, consisting of five ensemble members:



             

                           Member 5
      Member 2 --·           |
                 |           |
                 |           |
Member 1 ----·   |           |
             |   |           |
            \|/ \|/         \|/
           [ P0  P0  P0  P0  P0 ]
           [ P1  P1  P1  P1  P1 ]
           [ P2  P2  P2  P2  P2 ]
           [ P3  P3  P3  P3  P3 ]
           [ R0  R0  R0  R0  R0 ]
       A = [ R1  R1  R1  R1  R1 ]
           [ R2  R2  R2  R2  R2 ]
           [ F0  F0  F0  F0  F0 ]
           [ F1  F1  F1  F1  F1 ]
           [ F2  F2  F2  F2  F2 ]
           [ F3  F3  F3  F3  F3 ]
        

The in memory the matrix will look like this:

                                     
                                      Member 2
                                          | 
                            ______________|______________  
                           /              |              \  
                           |              |              |
             ______________|______________|______________|______________
            /              |              |              |              \
            |              |              |              |              |
            |              |              |              |              | 
           \|/            \|/            \|/            \|/            \|/         ...........    
   A =[ P0 P0 P0 P0 P0 P1 P1 P1 P1 P1 P2 P2 P2 P2 P2 P3 P3 P3 P3 P3 R0 R0 R0 R0 R0 R1 R1 R1 R1 R1 R2 R2 R2 R2 R2 F0 F0 F0 F0 F0 F1 F1 F1 F1 F1 F2 F2 F2 F2 F2 F3 F3 F3 F3 F3 ...]
       /|\    X1      /|\    X2      /|\            /|\            /|\              ........
        |              |              |              |              |
        |              |              |              |              |
        \______________|______________|______________|______________/
                       |              |              | 
                       |              |              |
                       \______________|______________/
                                      |
                                      | 
                                  Member 1

The stride in the serial_vector_type object is the number of elements
between consecutive elements in the same member, i.e. it is five in
the vector above. (Starting at e.g. P0 for member three (marked with
X1 in the figure), P1 for the same member is five elements down in the
vector (marked with X2 above)). Now - that was clear ehhh?


subnodes
========
Now - there is even more to this gruesome interface. In most cases the
enkf_node objects will contain *ONE* (float *) or (double *) data
field, and that is what is serialized here. However, also at the node
level we can have data distributed in several fields. For instance the
equil_type could have been implemented as[1]:

  struct equil_struct {
     double * woc;
     double * goc;
  }

i.e. with seperate storage for woc and goc. Then we would have to call
serialize two times, this must be done with the enkf_serialize_part()
and enkf_deserialize_part() functions. Internally the serialize code
use the concept node_index, that index operates on a (not existing)
vector of total length length(woc) + length(goc).



[1]: Currently the equil_strict is NOT implemented as this, but ....

/*****************************************************************/


/**
   This is a struct holding the serialized vector, along with some
   meta information. Observe that this struct will be read by many
   threads concurrently, it can therefor only hold static information,
   and *NOT* any pr. node or pr. member information.
*/

struct serial_vector_struct {
  double * serial_data;        /* The actual serialized storage - the ensemble matrix. */
  size_t   serial_size;        /* The size of serial_vector. */
  int      serial_stride;      /* The stride in the serial vector. See documentation of stride above.*/
};




/**
   This is a support struct designed to hold information about a
   partly completed serialization. Observe that there are two reasons
   why the serialization of one object might be called repeatedly:

    * The serial vector is full - i.e. we run out of memory.  We are
    * using local analysis, and call several times to update different
      parts of the object.

*/

struct serial_state_struct {
  int        node_index1;          /* The first node_index which is currently serialized. */
  int        node_index2;          /* The last node_index which is (not) currently serialized. */


  /* The fields current_node_index and serial_index are only needed to support subnodes, 
     i.e. explicit use of the xxx_part() functions. */
  int        current_node_index;   /* The node index we aer currently looking at - holding state between subsequetn calls to xx_part(). */
  size_t     serial_index;         /* Have to remember serial index (after offset) when subnodes are used. */
  
  size_t     serial_offset;        /* The current offset into the serial vector. */
  bool       complete;             /* Whether we are done with serialization of this node. */
  state_enum state;                /* enum which says which 'state' : {forecast, serialized, analyzed} the object is currently in. */
};

/*****************************************************************/


/**
  Observe that target_serial_size and _serial_size count the number of
  double elements, *NOT* the number of bytes, hence it is essential to
  have a margin to avoid overflow of the size_t datatype (on 32 bit
  machines).
  
*/

serial_vector_type * serial_vector_alloc(size_t target_serial_size, int ens_size) {
  size_t   serial_size = target_serial_size;
  double * serial_data;

#ifdef i386
  /* 
     33570816 = 2^25 is the maximum number of doubles we will
     allocate, this corresponds to 2^28 bytes - which it seems
     we can adress quite safely ...
  */
  serial_size = util_int_min(serial_size , 33570816 ); 
#endif
  
  do {
    serial_data = malloc(serial_size * sizeof * serial_data);
    if (serial_data == NULL) 
      serial_size /= 2;

  } while (serial_data == NULL);

  
  /*
    Ensure that the allocated memory is an integer times ens_size.
  */
  {
    int serial_size0 = serial_size;
    {
      div_t tmp   = div(serial_size , ens_size);
      serial_size = ens_size * tmp.quot;
    }
    if (serial_size != serial_size0) {
      /* 
	 Can not use realloc() here because the temporary memory
         requirements might be prohibitive, and furthermore memory
         copy is not really interesting.
      */
      free(serial_data);
      serial_data = util_malloc(serial_size * sizeof * serial_data , __func__);
    }
  }
  {
    serial_vector_type * serial_vector = util_malloc(sizeof * serial_vector , __func__);

    serial_vector->serial_data   = serial_data;
    serial_vector->serial_size   = serial_size;
    serial_vector->serial_stride = ens_size;

    return serial_vector;
  }
}


int serial_vector_get_stride(const serial_vector_type * serial_vector) {
  return serial_vector->serial_stride;
}

double * serial_vector_get_data(const serial_vector_type * serial_vector) {
  return serial_vector->serial_data;
}


void serial_vector_free(serial_vector_type * serial_vector) {
  free( serial_vector->serial_data );
  free( serial_vector );
}


/*****************************************************************/

void serial_state_clear(serial_state_type * state) {
  state->node_index1         = 0;
  state->node_index2         = 0;
  state->serial_offset       = 0;
  state->current_node_index  = 0;
  
  state->complete            = false;
  state->state               = forecast;
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
    Observe: The first eight  arguments regard the node which is
    currently being serialized, whereas the last four are about the
    serial vector which will hold the serialized data.
*/

size_t enkf_serialize_part(const void * __node_data         ,   /* The data of the input node - pointer to either float or double */
			   bool          first_call         ,   /* Always TRUE - except when this is the second++ subnode. */
			   int           node_size          ,   /* The size (number of elements) of __node_data array. */
			   int           node_offset        ,   /* The internal offset into the multicompnent node - zero for all simple nodes. */
			   int           total_node_size    , 
			   ecl_type_enum node_type          ,   /* The underlying data type of __node_data. */
			   const bool * active              ,   /* An active flag for the data - can be NULL if everything is active. */      
			   serial_state_type * serial_state ,   /* Holding the state of the current serialization of this node. */
			   /*-- Above: node data --- Below: serial data --*/
			   size_t serial_offset ,               /* The offset in the serial vector we are starting on - owned by the node; pointing into the serial vector. */                     
			   serial_vector_type * serial_vector) {

  int    node_index1        = serial_state->node_index1;
  int    current_node_index;
  size_t node_index         = node_index1;
  size_t elements_added     = 0;

  if (first_call) {
    serial_state->serial_index = 0;
    serial_state->current_node_index = node_index1;
  }
  current_node_index = serial_state->current_node_index;

  printf("%s:  first_call:%d \n",__func__ , first_call);

  if (!serial_state->complete) {
    if (node_index1 < node_offset + node_size) {
      /* If the previous _exactly_ used all memory we will have serial_size == serial_offset - and should return immediatebly. */
      if (serial_offset < serial_vector->serial_size) {  
	printf("current_node_index:%d node_index:%d  node_offset:%d  node_size:%d   serial_state->serial_index:%d \n",current_node_index , node_index , node_offset , node_size , serial_state->serial_index);
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
	printf("node_index:%d  node_offset:%d  node_size:%d \n",node_index , node_offset , node_size);
	if (node_index < (node_offset + node_size - 1)) 
	  /* We did not get through the complete object after all ... */
	  serial_state->node_index2 = node_index + 1;
	else {
	  /* We made it all the way through the object. */ 
	  serial_state->node_index2 = node_offset + node_size;
	  if (serial_state->node_index2 == total_node_size) {
	    /* We are completely done with this node - including all subnodes. */ 
	    serial_state->state            = serialized;
	    serial_state->complete         = true;
	  }
	}
	serial_state->current_node_index = node_index ;
	printf("Have serialized: %d - %d \n",serial_state->node_index1 , serial_state->node_index2);
	serial_state->serial_offset    = serial_offset;
      }
    }
  }
  return elements_added;
}


/**
   This function should be used in the default case where a enkf_node
   object only contains one (double *) / (float *) pointer which
   should be serialized.

   It then calls enkf_serialize_part() with some defaulted arguments.
*/


size_t enkf_serialize(const void * __node_data         ,  
                      int           node_size          ,  
		      ecl_type_enum node_type          ,  
                      const bool * active              ,  
                      serial_state_type * serial_state ,  
                      size_t serial_offset ,              
                      serial_vector_type * serial_vector) {
  
  bool first_call      = true;
  int  node_offset     = 0;
  int  total_node_size = node_size;

  enkf_serialize_part(__node_data , first_call , node_size , node_offset , total_node_size , node_type , active , serial_state , serial_offset , serial_vector);
}




void enkf_deserialize_part(void * __node_data                , /* The data of the node which will accept updated data. */
			   bool     first_call               , /* Always TRUE - except when this is the second++ subnode. */
			   int      node_size                , /* The total size of the node we are updating. */
			   int      node_offset              , /* The current offset into the node i.e. how many elements we have serialized. */            
			   int      total_node_size          , /* The TOTAL size of the node (including inactive ++) */       
			   ecl_type_enum node_type           , /* The underlying type (double || float) of the node's storage. */
			   const bool * active               , /* Active / inactive flag - can be NULL for all elements active. */
			   serial_state_type * serial_state  , /* Holding the state of the current serialization of this node. */
			   /*-- Above: node data ---  Below: serial data --*/
			   const serial_vector_type * serial_vector) {
  
  size_t serial_offset = serial_state->serial_offset;
  int    node_index1   = serial_state->node_index1;
  int    node_index2   = serial_state->node_index2;
  int    current_node_index;
  if (first_call) {
    serial_state->serial_index = 0;
    serial_state->current_node_index = node_index1;
  }
  current_node_index = serial_state->current_node_index;

  if (serial_state->state == serialized) {
    if (node_index1 < node_offset + node_size) {  /* The node we are looking at contains, or follows after, node_index1 */
      
      if (node_type == ecl_double_type) {
	double * node_data = (double *) __node_data;
#include "deserialize.h"
      } else if (node_type == ecl_float_type) {
	float * node_data = (float *) __node_data;
#include "deserialize.h"
      } else 
	util_abort("%s: internal error: trying to deserialize unserializable type:%s \n",__func__ , ecl_util_type_name( node_type ));
      
      if (serial_state->complete) 
	serial_state->state        = analyzed;
      else 
	serial_state->node_index1  = serial_state->node_index2;  /* This is where the next serialization should start. */
    }
  }
}



void enkf_deserialize(void * __node_data                , 
		      int      node_size                , 
		      ecl_type_enum node_type           , 
		      const bool * active               , 
		      serial_state_type * serial_state  , 
		      const serial_vector_type * serial_vector) {

  bool first_call      = true;
  int  node_offset     = 0;
  int  total_node_size = node_size;
  
  enkf_deserialize_part(__node_data , first_call , node_size , node_offset , total_node_size , node_type , active , serial_state , serial_vector);
}
