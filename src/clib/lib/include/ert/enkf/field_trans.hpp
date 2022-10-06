#ifndef ERT_FIELD_TRANS_H
#define ERT_FIELD_TRANS_H
#include <stdbool.h>
#include <stdio.h>

typedef float(field_func_type)(float);
typedef struct field_trans_table_struct field_trans_table_type;

void field_trans_table_fprintf(const field_trans_table_type *, FILE *);
void field_trans_table_free(field_trans_table_type *);
void field_trans_table_add(field_trans_table_type *, const char *, const char *,
                           field_func_type *);
field_trans_table_type *field_trans_table_alloc();
bool field_trans_table_has_key(field_trans_table_type *, const char *);
field_func_type *field_trans_table_lookup(field_trans_table_type *,
                                          const char *);

#endif
