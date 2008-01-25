#ifndef __SET_H__
#define __SET_H__

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include <hash.h>

typedef struct set_struct set_type;
set_type   * set_alloc(int , const char ** );
set_type   * set_alloc_empty();
bool         set_add_key(set_type * , const char * );
void         set_free(set_type * );
int          set_get_size(const set_type *);
char      ** set_alloc_keylist(const set_type * );



#endif
