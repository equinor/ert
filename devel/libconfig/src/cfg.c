#include <list.h>

typedef struct cfg_node_struct cfg_node_type;
typedef struct cfg_struct      cfg_type;

struct cfg_node_struct
{
  char * key;
  char * value;
  list_type * children;   /* Children are cfg_node_struct's. */
};



struct cfg_struct
{
  list_type * roots;
};
