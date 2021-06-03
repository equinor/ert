/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'arg_pack.c' is part of ERT - Ensemble based Reservoir Tool.

   ERT is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   ERT is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or
   FITNESS FOR A PARTICULAR PURPOSE.

   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
   for more details.
*/

#include <stdlib.h>
#include <string.h>

#include <ert/util/util.h>
#include <ert/util/node_ctype.hpp>

#include <ert/res_util/arg_pack.hpp>


/**
   This file implements an arg_pack structure which is a small
   convienence utility to pack several arguments into one
   argument. The generic use situtation is when calling functions like
   e.g. pthread_create() which take one (void *) as argument. You can
   then pack several arguments into one arg_pack instance, and then
   unpack them at the other end.

   The content of the arg_pack is mainly inserted by appending, in
   addition it is possible to insert new items by using _iset()
   functions, however these functions will fail hard if the resulting
   call sequence will lead to holes in the structure, i.e.

      arg_pack_type * arg_pack = arg_pack_alloc()
      arg_pack_append_int( arg_pack , 1);

   When you take them out again that is done with indexed get.

   When elements are inserted into the arg_pack, they are inserted
   with a (limited) type information (implictly given by the function
   invoked to insert the argument), and the corresponding typed get
   must be used to unpack the argument again afterwards. The
   excepetion is with the function arg_pack_iget_adress() which can be
   used to extract the reference of a scalar.



   Example:
   --------

   void some_function(const char * arg1 , int arg2 , double arg3) {
      .....
   }


   void some_function__(void * __arg_pack) {
      arg_pack_type * arg_pack = arg_pack_safe_cast( __arg_pack );
      const char * arg1 = arg_pack_iget_ptr( arg_pack , 0);
      int          arg2 = arg_pack_iget_int( arg_pack , 1);

      some_function( arg1 , arg2 , arg3 );
   }


   .....
   arg_pack_type * arg_pack = arg_pack_alloc();
   arg_pack_append_ptr(arg_pack , "ARG1");
   arg_pack_append_int(arg_pack , 1);
   arg_pack_append_double(arg_pack , 3.14159265);

   pthread_create( ,  , some_function__ , arg_pack);

*/




#define ARG_PACK_TYPE_ID 668268


typedef struct {
  void                 * buffer;        /* This is the actual content - can either point to a remote object, or to storage managed by the arg_pack instance. */
  node_ctype             ctype;         /* The type of the data which is stored. */
  arg_node_free_ftype  * destructor;    /* destructor called on buffer - can be NULL. */
  arg_node_copyc_ftype * copyc;         /* copy constructor - will typically be NULL. */
} arg_node_type;



struct arg_pack_struct {
  UTIL_TYPE_ID_DECLARATION;
  int             size;                /* The number of arguments appended to this arg_pack instance. */
  int             alloc_size;          /* The number of nodes allocated to this arg_pack - will in general be greater than size. */
  bool            locked;              /* To insure against unwaranted modifictaions - you can explicitly lock the arg_pack instance. This only */
  arg_node_type **nodes;               /* Vector of nodes */
};


/*****************************************************************/
/* First comes the arg_node functions. These are all fully static.*/

static arg_node_type * arg_node_alloc_empty() {
  arg_node_type * node = (arg_node_type*)util_malloc( sizeof * node );
  node->buffer      = NULL;
  node->destructor  = NULL;
  node->ctype       = CTYPE_INVALID;
  return node;
}


static void arg_node_realloc_buffer(arg_node_type * node , int new_size) {
  node->buffer      = util_realloc(node->buffer , new_size );
}


static void __arg_node_assert_type(const arg_node_type * node , node_ctype arg_type) {
  if (arg_type != node->ctype)
    util_abort("%s: asked for type:\'%s\'  inserted as:\'%s\'  - aborting \n" , __func__ , node_ctype_name(arg_type) , node_ctype_name(node->ctype));
}



/*****************************************************************/
#define ARG_NODE_GET_RETURN(type)                \
{                                                \
   type value;                                   \
   memcpy(&value , node->buffer , sizeof value); \
   return value;                                 \
}

// Used by IGET_TYPED macro
C_USED static int arg_node_get_int( const arg_node_type * node) {
  __arg_node_assert_type( node , CTYPE_INT_VALUE );
  ARG_NODE_GET_RETURN( int )
}

C_USED static char arg_node_get_char( const arg_node_type * node) {
  __arg_node_assert_type( node , CTYPE_CHAR_VALUE );
  ARG_NODE_GET_RETURN( char )
}

C_USED static double arg_node_get_double( const arg_node_type * node) {
  __arg_node_assert_type( node , CTYPE_DOUBLE_VALUE );
  ARG_NODE_GET_RETURN( double )
}

C_USED static float arg_node_get_float( const arg_node_type * node) {
  __arg_node_assert_type( node , CTYPE_FLOAT_VALUE );
  ARG_NODE_GET_RETURN( float )
}

C_USED static bool arg_node_get_bool( const arg_node_type * node) {
  __arg_node_assert_type( node , CTYPE_BOOL_VALUE );
  ARG_NODE_GET_RETURN( bool )
}

C_USED static size_t arg_node_get_size_t( const arg_node_type * node) {
  __arg_node_assert_type( node , CTYPE_SIZE_T_VALUE );
  ARG_NODE_GET_RETURN( size_t )
}
#undef ARG_NODE_GET_RETURN

/**
   If the argument is inserted as a pointer, you must use get_ptr ==
   true, otherwise you must use get_ptr == false, and this will give
   you the adress of the scalar.

   Observe that if you call XX_get_ptr() on a pointer which is still
   owned by the arg_pack, you must be careful when freeing the
   arg_pack, as that will delete the pointer you are using as well.
*/


static void * arg_node_get_ptr(const arg_node_type * node , bool get_ptr) {
  if (get_ptr) {
    if (node->ctype != CTYPE_VOID_POINTER)
      util_abort("%s: tried to get pointer from something not a pointer\n",__func__);
  } else {
    if (node->ctype == CTYPE_VOID_POINTER)
      util_abort("%s: tried to get adress to something already a ponter\n",__func__);
  }
  return node->buffer;
}


static node_ctype arg_node_get_ctype( const arg_node_type * arg_node ) {
  return arg_node->ctype;
}

/*****************************************************************/
/* SET functions. */


#define ARG_NODE_SET(node , value)                                     \
  arg_node_realloc_buffer(node , sizeof value);                        \
  memcpy(node->buffer , &value , sizeof value);                        \
  node->destructor = NULL;

// Used by ISET_TYPED macro
C_USED static void arg_node_set_int( arg_node_type * node , int value) {
  ARG_NODE_SET( node , value );
  node->ctype = CTYPE_INT_VALUE;
}


C_USED static void arg_node_set_char( arg_node_type * node , char value) {
  ARG_NODE_SET( node , value );
  node->ctype = CTYPE_CHAR_VALUE;
}


C_USED static void arg_node_set_float( arg_node_type * node , float value) {
  ARG_NODE_SET( node , value );
  node->ctype = CTYPE_FLOAT_VALUE;
}


C_USED static void arg_node_set_double( arg_node_type * node , double value) {
  ARG_NODE_SET( node , value );
  node->ctype = CTYPE_DOUBLE_VALUE;
}


C_USED static void arg_node_set_bool( arg_node_type * node , bool value) {
  ARG_NODE_SET( node , value );
  node->ctype = CTYPE_BOOL_VALUE;
}


C_USED static void arg_node_set_size_t( arg_node_type * node , size_t value) {
  ARG_NODE_SET( node , value );
  node->ctype = CTYPE_SIZE_T_VALUE;
}

#undef ARG_NODE_SET


static void arg_node_set_ptr(arg_node_type * node , const void * ptr , arg_node_copyc_ftype * copyc , arg_node_free_ftype * destructor) {
  node->ctype      = CTYPE_VOID_POINTER;
  node->destructor = destructor;
  node->copyc      = copyc;
  if (copyc != NULL)
    node->buffer = copyc( ptr );
  else
    node->buffer = (void *) ptr;
}



/*****************************************************************/


static void arg_node_clear(arg_node_type * node) {
  if (node->ctype == CTYPE_VOID_POINTER) {
    if (node->destructor != NULL)
      node->destructor( node->buffer );
    /* When you have cleared - must not reuse the thing. */
    node->destructor = NULL;
    node->buffer     = NULL;
    node->copyc      = NULL;
  }
}


static void arg_node_free(arg_node_type * node) {
  arg_node_clear(node);
  free(node->buffer);
  free(node);
}

/*****************************************************************/
/* Ending node node functions - starting on functons for the whole pack. */
/*****************************************************************/

UTIL_SAFE_CAST_FUNCTION( arg_pack , ARG_PACK_TYPE_ID)
UTIL_SAFE_CAST_FUNCTION_CONST( arg_pack , ARG_PACK_TYPE_ID)
UTIL_IS_INSTANCE_FUNCTION(arg_pack , ARG_PACK_TYPE_ID)

static void __arg_pack_assert_index(const arg_pack_type * arg , int iarg) {
  if (iarg < 0 || iarg >= arg->size)
    util_abort("%s: arg_pack() object filled with %d arguments - %d invalid argument number - aborting \n",__func__ , arg->size , iarg);
}


static void arg_pack_realloc_nodes(arg_pack_type * arg_pack , int new_size) {
  arg_pack->nodes      = (arg_node_type**) util_realloc(arg_pack->nodes , new_size * sizeof * arg_pack->nodes );
  {
    int i;
    for (i = arg_pack->alloc_size; i < new_size; i++)
      arg_pack->nodes[i] = arg_node_alloc_empty();
  }
  arg_pack->alloc_size = new_size;
}


/**
   The name of this function is QUITE MISLEADING; the function will
   create a new node, with index @index, and return it possibly
   freeing the existing with this index. If index == arg_pack->size a
   new node will be created at the end of the arg_pack; if index >
   arg_pack->size the function will fail hard.
*/
static arg_node_type * arg_pack_iget_new_node( arg_pack_type * arg_pack , int index) {
  if (index < 0 || index > arg_pack->size)
    util_abort("%s: index:%d invalid. Valid interval: [0,%d) \n",__func__ , index , arg_pack->size);
  {
    if (index < arg_pack->size) {
      arg_node_free( arg_pack->nodes[index] );           /* Free the existing current node. */
      arg_pack->nodes[index] = arg_node_alloc_empty( );  /* Allocate a new fresh instance. */
    }

    if (arg_pack->size == arg_pack->alloc_size)
      arg_pack_realloc_nodes(arg_pack , 1 + arg_pack->alloc_size * 2);  /* We have to grow the vector of nodes. */

    if (index == arg_pack->size)
      arg_pack->size++;            /* We are asking for the first element beyond the current length of the vector, i.e. append. */
    return arg_pack->nodes[index];
  }
}



static arg_node_type * arg_pack_get_append_node(arg_pack_type * arg_pack) {
  if (arg_pack->locked) {
    util_abort("%s: tryng to append to a locked arg_pack instance \n",__func__);
    return NULL;
  }
  {
    arg_node_type * new_node = arg_pack_iget_new_node( arg_pack , arg_pack->size );
    return new_node;
  }
}


arg_pack_type * arg_pack_alloc() {
  arg_pack_type * arg_pack = (arg_pack_type*)util_malloc(sizeof * arg_pack );
  UTIL_TYPE_ID_INIT( arg_pack , ARG_PACK_TYPE_ID);
  arg_pack->nodes      = NULL;
  arg_pack->alloc_size = 0;
  arg_pack->locked     = false;
  arg_pack_realloc_nodes(arg_pack , 4);
  arg_pack_clear(arg_pack);
  return arg_pack;
}


void arg_pack_free(arg_pack_type * arg_pack) {
  int i;

  for (i=0; i < arg_pack->alloc_size; i++)
    arg_node_free( arg_pack->nodes[i] );

  free(arg_pack->nodes);
  free(arg_pack);
}


void arg_pack_clear(arg_pack_type * arg_pack) {
  if (arg_pack->locked)
    util_abort("%s: arg_pack has been locked - abortng \n",__func__);
  {
    int i;
    for ( i=0; i < arg_pack->size; i++)
      arg_node_clear(arg_pack->nodes[i]);
    arg_pack->size = 0;
  }
}


/******************************************************************/
/* Access functions:

  1. Append
  2. iget
  3. iset (can NOT create holes in the vector)

******************************************************************/

#define APPEND_TYPED(type)                                         \
void arg_pack_append_ ## type (arg_pack_type *pack , type value) { \
  arg_node_type * node = arg_pack_get_append_node( pack );         \
  arg_node_set_ ## type(node , value);                             \
}


#define ISET_TYPED(type)\
void arg_pack_iset_ ## type(arg_pack_type * pack, int index, type value) {   \
  arg_node_type * node = arg_pack_iget_new_node( pack , index);  \
  arg_node_set_ ## type(node , value);                           \
}


#define IGET_TYPED(type)\
type arg_pack_iget_ ## type(const arg_pack_type * pack, int index) { \
  __arg_pack_assert_index( pack , index);                            \
  {                                                                  \
    arg_node_type * node = pack->nodes[index];                       \
    return arg_node_get_ ## type ( node );                           \
  }                                                                  \
}


APPEND_TYPED(int);
APPEND_TYPED(bool);
APPEND_TYPED(float);
APPEND_TYPED(double);
APPEND_TYPED(char);
APPEND_TYPED(size_t);

IGET_TYPED(int);
IGET_TYPED(bool);
IGET_TYPED(float);
IGET_TYPED(double);
IGET_TYPED(char);
IGET_TYPED(size_t);

ISET_TYPED(int);
ISET_TYPED(bool);
ISET_TYPED(float);
ISET_TYPED(double);
ISET_TYPED(char);
ISET_TYPED(size_t);

#undef APPEND_TYPED
#undef IGET_TYPED
#undef ISET_TYPED


void * arg_pack_iget_ptr(const arg_pack_type * arg , int iarg) {
  __arg_pack_assert_index(arg , iarg);
  return arg_node_get_ptr(arg->nodes[iarg] , true);
}


const void * arg_pack_iget_const_ptr(const arg_pack_type * arg , int iarg) {
  __arg_pack_assert_index(arg , iarg);
  return arg_node_get_ptr(arg->nodes[iarg] , true);
}


void * arg_pack_iget_adress(const arg_pack_type * arg , int iarg) {
  __arg_pack_assert_index(arg , iarg);
  return arg_node_get_ptr(arg->nodes[iarg] , false);
}


/*****************************************************************/


void  arg_pack_iset_copy(arg_pack_type * arg_pack , int index , const void * ptr, arg_node_copyc_ftype * copyc , arg_node_free_ftype * freef) {
  arg_node_type * node = arg_pack_iget_new_node( arg_pack , index );
  arg_node_set_ptr(node , ptr , copyc , freef);
}


void arg_pack_iset_ptr(arg_pack_type * arg_pack, int index , const void * ptr) {
  arg_pack_iset_copy(arg_pack , index , ptr , NULL , NULL);
}

void arg_pack_iset_owned_ptr(arg_pack_type * arg_pack, int index , void * ptr, arg_node_free_ftype * freef) {
  arg_pack_iset_copy(arg_pack , index , ptr , NULL , freef );
}

void arg_pack_append_ptr(arg_pack_type * arg_pack, void * ptr) {
  arg_pack_iset_ptr( arg_pack , arg_pack->size , ptr );
}

void arg_pack_append_const_ptr(arg_pack_type * arg_pack, const void * ptr) {
  arg_pack_iset_ptr( arg_pack , arg_pack->size , ptr );
}


void arg_pack_append_owned_ptr(arg_pack_type * arg_pack, void * ptr, arg_node_free_ftype * freef) {
  arg_pack_iset_owned_ptr( arg_pack , arg_pack->size , ptr , freef);
}

int arg_pack_size( const arg_pack_type * arg_pack ) {
  return arg_pack->size;
}
