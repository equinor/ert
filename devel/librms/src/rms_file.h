#ifndef __RMS_FILE_H__
#define __RMS_FILE_H__

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <rms_tag.h>
#include <list.h>


typedef struct rms_file_struct   rms_file_type;

void                 rms_file_2eclipse(const char * rms_file , const char * , bool , bool , int );
list_node_type     * rms_file_iterate_tag_node(const rms_file_type * , list_node_type * );
void                 rms_file_fclose(rms_file_type *);
FILE               * rms_file_fopen_r(rms_file_type *);
FILE               * rms_file_fopen_w(rms_file_type *);
void                 rms_file_set_filename(rms_file_type * , const char * , bool);
rms_file_type      * rms_file_alloc  	  (const char *, bool );
void                 rms_file_fread  	  (rms_file_type *);
void                 rms_file_fwrite 	  (rms_file_type * , const char *);
void                 rms_file_printf 	  (const rms_file_type * , FILE *);
void                 rms_file_free   	  (rms_file_type *);
void                 rms_file_free_data   (rms_file_type *);
rms_tag_type       * rms_file_get_dim_tag_ref(const rms_file_type * );
rms_tag_type       * rms_file_get_tag_ref (const rms_file_type *, const char *, const char *, const char * , bool);
void                 rms_file_assert_dimensions(const rms_file_type *, int , int , int );
rms_tag_type       * rms_file_fread_alloc_tag(rms_file_type * , const char *, const char *, const char *);
rms_tagkey_type    * rms_file_fread_alloc_data_tagkey(rms_file_type * , const char *, const char *, const char *);
void 		     rms_file_complete_fwrite(const rms_file_type *);
void 		     rms_file_init_fwrite(const rms_file_type * , const char *);
void                 rms_file_get_dims(const rms_file_type * , int * );
FILE               * rms_file_get_FILE(const rms_file_type * );
void                 rms_file_add_dimensions(rms_file_type * , int , int , int , bool);
bool                 rms_file_is_roff(FILE * );

#endif
