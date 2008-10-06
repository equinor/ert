#include <string.h>
#include <hash.h>
#include <util.h>
#include <stringlist.h>
#include <cfg_util.h>
#include <cfg_lang.h>

typedef struct cfg_kw_def_struct cfg_kw_def_type;



typedef enum {UNIQ, DATA, REQUIRE, KEY, RESTRICT} cfg_kw_keys_enum;
// These are the primitive keys which are allowed in the definition.
#define UNIQ_STRING     "unique"
#define DATA_STRING     "data"
#define REQUIRE_STRING  "require"
#define KEY_STRING      "key"
#define RESTRICT_STRING "restrict"



typedef enum {CFG_DATA_INT, CFG_DATA_POSINT, CFG_DATA_FLOAT, CFG_DATA_POSFLOAT, CFG_DATA_FILE, CFG_DATA_DATE} cfg_kw_data_enum;
// These are types that we can require a value to validate as.
//
// CFG_INT      - Integer.
// CFG_POSINT   - Positive integer.
// CFG_FLOAT    - Floating point number.
// CFG_POSFLOAT - Positive floating point number.
// CFG_FILE     - Filename to a file that can be read.
// CFG_DATE     - A date on the form DD-MM-YYYY.
#define CFG_DATA_INT_STRING      "int"
#define CFG_DATA_POSINT_STRING   "posint"
#define CFG_DATA_FLOAT_STRING    "float"
#define CFG_DATA_POSFLOAT_STRING "posfloat"
#define CFG_DATA_FILE_STRING     "file"
#define CFG_DATA_DATE_STRING     "date"



typedef enum {CFG_UNIQ_NO, CFG_UNIQ_KEY, CFG_UNIQ_KEY_OW, CFG_UNIQ_KEYVAL, CFG_UNIQ_KEYVAL_OW}   cfg_kw_uniq_enum;
// Actions taken if key, or key value combination is encountered
// several times in the same scope.
//
// NO        -  No action.
// KEY       -  Abort if the key has been set.
// KEY_OW    -  Overwrite the previous instance of the key.
// KEYVAL    -  Abort if the key and value has been set.
// KEYVAL_OW -  Overwrite the previous instance of key and value.
#define CFG_UNIQ_NO_STRING        "no"
#define CFG_UNIQ_KEY_STRING       "key"
#define CFG_UNIQ_KEY_OW_STRING    "key_ow"
#define CFG_UNIQ_KEYVAL_STRING    "keyval"
#define CFG_UNIQ_KEYVAL_OW_STRING "keyval_ow"



#define RETURN_IF_MATCH(TYPE) if(strcmp(str, TYPE ##_STRING) == 0){ return TYPE;}
static cfg_kw_keys_enum get_cfg_kw_keys_enum_from_string(const char * str)
{
  RETURN_IF_MATCH(UNIQ);
  RETURN_IF_MATCH(DATA);
  RETURN_IF_MATCH(REQUIRE);
  RETURN_IF_MATCH(KEY);
  RETURN_IF_MATCH(RESTRICT);

  util_abort("%s: Key %s is unkown.\n", __func__, str);
  return 0;
}



static cfg_kw_data_enum get_cfg_kw_data_enum_from_string(const char * str)
{
  RETURN_IF_MATCH(CFG_DATA_INT);
  RETURN_IF_MATCH(CFG_DATA_POSINT);
  RETURN_IF_MATCH(CFG_DATA_FLOAT);
  RETURN_IF_MATCH(CFG_DATA_POSFLOAT);
  RETURN_IF_MATCH(CFG_DATA_FILE);
  RETURN_IF_MATCH(CFG_DATA_DATE);

  util_abort("%s: Data type %s is unkown.\n", __func__, str);
  return 0;
}



static cfg_kw_uniq_enum get_cfg_kw_uniq_enum_from_string(const char * str)
{
  RETURN_IF_MATCH(CFG_UNIQ_NO);
  RETURN_IF_MATCH(CFG_UNIQ_KEY);
  RETURN_IF_MATCH(CFG_UNIQ_KEY_OW);
  RETURN_IF_MATCH(CFG_UNIQ_KEYVAL);
  RETURN_IF_MATCH(CFG_UNIQ_KEYVAL_OW);

  util_abort("%s: Uniqueness type %s is unkown.\n", __func__, str);
  return 0;
}
#undef RETURN_IF_MATCH



struct cfg_kw_def_struct{
  char * key;
  cfg_kw_uniq_enum uniq;

  cfg_kw_data_enum   type; 
  stringlist_type  * restriction_set;

  /* Children are cfg_kw_def_struct's. */
  hash_type       * children;

 /* Require these child kw's to be set. */
  stringlist_type * required_children;
};



struct cfg_lang_def_struct{
  /* A hash fo cfg_kw_def_struct's indexed by key's. */
  hash_type * roots;
};



void cfg_kw_def_free(cfg_kw_def_type * cfg_kw_def)
{
  free(           cfg_kw_def->key);
  hash_free(      cfg_kw_def->children);
  stringlist_free(cfg_kw_def->required_children);

}



void cfg_kw_def_free__(void * cfg_kw_def)
{
  cfg_kw_def_free( (cfg_kw_def_type *) cfg_kw_def);
}



cfg_kw_def_type * cfg_kw_def_fscanf_alloc(FILE * stream, bool * at_eof, const char * key)
{
  bool uniq_set  = false;
  bool type_set  = false;

  cfg_kw_def_type * cfg_kw_def  = util_malloc(sizeof * cfg_kw_def, __func__);

  cfg_kw_def->key               = util_alloc_string_copy(key);
  cfg_kw_def->children          = hash_alloc();
  cfg_kw_def->required_children = stringlist_alloc_new();
  cfg_kw_def->restriction_set   = stringlist_alloc_new();


  bool sub_start = false;
  bool sub_end   = false;

  *at_eof    = false;

  char * token1   = NULL;
  char * token2   = NULL;
  
  /*
    Read until we encounter '{' or a syntax error.
  */
  while(!*at_eof && !sub_end && !sub_start && token1 == NULL)
     token1 = cfg_util_fscanf_alloc_token(stream, &sub_start, &sub_end, at_eof);

  if(sub_end)
    util_abort("%s: Unexpected end of keyword definition %s.\n", __func__, key);
  if(*at_eof)
    util_abort("%s: File ended before definition of keyword %s.\n", __func__, key);
  if(token1 != NULL)
    util_abort("%s: Expected '{', got %s.\n", __func__, token1);

  while(!*at_eof && !sub_end)
  {
    /*
      Read a key value combination.
    */
    while(!*at_eof && !sub_end && !sub_start && token1 == NULL)
       token1 = cfg_util_fscanf_alloc_token(stream, &sub_start, &sub_end, at_eof);

    if(sub_end && (!uniq_set || !type_set))
      util_abort("%s: Unexpected end of keyword definition %s.\n", __func__, key);
    else if(sub_end)
      break;

    if(*at_eof)
      util_abort("%s: File ended before definition of keyword %s.\n", __func__, key);
    if(sub_start)
      util_abort("%s: Unexpected '{' %s.\n", __func__, token1);
  
    /**
      Ok, we have read a valid key. Now read the value. 
      
      Should maybe allow to read "help" kw here?
    */

    token2 = cfg_util_fscanf_alloc_token(stream, &sub_start, &sub_end, at_eof);

    if(sub_end)
      util_abort("%s: Unexpected end of keyword definition %s. Read key %s without value.\n", __func__, key, token1);
    if(*at_eof)
      util_abort("%s: File ended before definition of keyword %s.\n", __func__, key);
    if(sub_start)
      util_abort("%s: Unexpected '{' %s.\n", __func__, token1);

    /* token1 and token2 are now a valid key/value pair. */

    cfg_kw_keys_enum cfg_kw_key  = get_cfg_kw_keys_enum_from_string(token1);

    switch(cfg_kw_key)
    {
      case(UNIQ):
      {
        cfg_kw_def->uniq = get_cfg_kw_uniq_enum_from_string(token2);
        uniq_set = true;
        break;
      }
      case(REQUIRE):
      {
        stringlist_append_copy(cfg_kw_def->required_children, token2);
        break;
      }
      case(DATA):
      {
       cfg_kw_def->type = get_cfg_kw_data_enum_from_string(token2); 
       type_set = true;
       break;
      }
      case(KEY):
      {
        cfg_kw_def_type * cfg_kw_def_child = cfg_kw_def_fscanf_alloc(stream, at_eof, token2);
        hash_insert_hash_owned_ref(cfg_kw_def->children, token2, cfg_kw_def_child, cfg_kw_def_free__);
        break;
      }
    }
  }

  return cfg_kw_def;
}
