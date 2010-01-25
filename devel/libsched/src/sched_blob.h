#ifndef __SCHED_BLOB_H__
#define __SCHED_BLOB_H__
#ifdef __cplusplus 
extern "C" {
#endif
#include <stdlib.h>

typedef struct sched_blob_struct sched_blob_type;


void              sched_blob_append_token( sched_blob_type * blob , const char * token );
void              sched_blob_append_tokens( sched_blob_type * blob , const stringlist_type * tokens , int offset , int length );
sched_blob_type * sched_blob_alloc( );
void              sched_blob_free( sched_blob_type * blob );
void              sched_blob_fprintf( const sched_blob_type * blob , FILE * stream );
int               sched_blob_get_size( const sched_blob_type * blob );

#ifdef __cplusplus 
}
#endif
#endif
