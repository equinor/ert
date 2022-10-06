#ifndef ERT_FIELD_COMMON_H
#define ERT_FIELD_COMMON_H

/*
  Contains some headers which both field.c and field_config.c need -
  split like this to avoid circular dependencies.
*/

typedef struct field_config_struct field_config_type;
typedef struct field_struct field_type;

field_type *field_alloc(const field_config_type *);
bool field_fload(field_type *, const char *);

#endif
