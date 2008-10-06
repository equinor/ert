#include <hash.h>
#include <stringlist.h>

typedef enum {CFG_INT, CFG_POSINT, CFG_FLOAT, CFG_POSFLOAT, CFG_FILE, CFG_DATE} cfg_kw_data_enum;
// These are types that we can require a value to validate as.
//
// CFG_INT      - Integer.
// CFG_POSINT   - Positive integer.
// CFG_FLOAT    - Floating point number.
// CFG_POSFLOAT - Positive floating point number.
// CFG_FILE     - Filename to a file that can be read.
// CFG_DATE     - A date on the form DD-MM-YYYY.

typedef enum {NO, KEY, KEY_OW, KEYVAL, KEYVAL_OW}   cfg_kw_uniq_enum;
// Actions taken if key, or key value combination is encountered
// several times in the same scope.
//
// NO        -  No action.
// KEY       -  Abort if the key has been set.
// KEY_OW    -  Overwrite the previous instance of the key.
// KEYVAL    -  Abort if the key and value has been set.
// KEYVAL_OW -  Overwrite the previous instance of key and value.



struct cfg_kw_def_struct{
  char * key;
  cfg_kw_uniq_enum uniq;

  cfg_kw_data_enum   type; 
  stringlist_type  * restriction_set;

  hash_type       * children;          /* Children are cfg_kw_def_struct's. */
  stringlist_type * required_children; /* Require these child kw's to be set. */
};



struct cfg_lang_def_struct{
  hash_type * roots;
};



