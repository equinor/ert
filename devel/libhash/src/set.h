#ifndef __SET_H__
#define __SET_H__

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include <hash.h>

typedef struct set_struct set_type;
void         set_free_ext_keylist(const set_type * , char ** );
void         set_remove_key(set_type * , const char * );
set_type   * set_alloc(int , const char ** );
set_type   * set_alloc_empty();
bool         set_add_key(set_type * , const char * );
bool         set_has_key(set_type * set, const char * );
void         set_free(set_type * );
int          set_get_size(const set_type *);
char      ** set_alloc_keylist(const set_type * );
void         set_fwrite(const set_type * , FILE * );
void         set_fread(set_type * , FILE * );
set_type   * set_fread_alloc(FILE *);
void         set_fprintf(const set_type * , FILE * );


#endif
