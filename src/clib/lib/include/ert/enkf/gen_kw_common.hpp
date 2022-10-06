#ifndef ERT_GEN_KW_COMMON_H
#define ERT_GEN_KW_COMMON_H

/*
  Contains some headers which both gen_kw.c and gen_kw_config.c need -
  split like this to avoid circular dependencies.
*/

#include <ert/tooling.hpp>

typedef struct gen_kw_config_struct gen_kw_config_type;
typedef struct gen_kw_struct gen_kw_type;

extern "C" PY_USED gen_kw_type *gen_kw_alloc(const gen_kw_config_type *);

#endif
