#include <util.h>
#include <string.h>
#include <stringlist.h>
#include <sched_util.h>
#include <sched_blob.h>
#include <sched_time.h>
#include <buffer.h>
#include <time.h>





struct sched_blob_struct {
  char              * buffer;
  time_t              start_time;
  sched_time_type   * time_step;   /* Either a date into the 'future' - or a TSTEP. This is the end time of the blob.*/
};



static void sched_blob_append_buffer( sched_blob_type * blob , const char * new_buffer ) {
  blob->buffer = util_strcat_realloc( blob->buffer , new_buffer );
}


void sched_blob_append_token( sched_blob_type * blob , const char * token ) {
  char * new_buffer = util_malloc( (strlen(token) + 2) * sizeof * new_buffer , __func__);
  sched_blob_append_buffer( blob , new_buffer );
  free( new_buffer );
}


void sched_blob_append_tokens( sched_blob_type * blob , const stringlist_type * tokens , int offset , int length ) {
  char * new_buffer = stringlist_alloc_joined_segment_string( tokens , offset , length , " ");
  sched_blob_append_buffer( blob , new_buffer );
  free( new_buffer );
}


sched_blob_type * sched_blob_alloc() {
  sched_blob_type * blob = util_malloc( sizeof * blob , __func__);
  blob->buffer    = NULL;
  blob->time_step = NULL;
  return blob;
}


int sched_blob_get_size( const sched_blob_type * blob ) {
  if (blob->buffer == NULL)
    return 0;
  else
    return strlen( blob->buffer );
}



void sched_blob_free( sched_blob_type * blob ) {
  util_safe_free( blob->buffer );
  if (blob->time_step != NULL)
    sched_time_free( blob->time_step );
  free( blob );
}




void sched_blob_fprintf( const sched_blob_type * blob , FILE * stream ) {
  fprintf(stream , "%s" , blob->buffer );
}
