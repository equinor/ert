#ifndef __HISTORY_H__
#define __HISTORY_H__



#include <stdio.h>
#include <sched_file.h>



typedef struct history_struct history_type;



void           history_free(history_type *);
void           history_fwrite(const history_type *, FILE * stream);
history_type * history_fread_alloc(FILE * stream);
history_type * history_alloc_from_sched_file(const sched_file_type *);
#endif
