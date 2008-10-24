#include <assert.h>
#include <string.h>
#include <stdbool.h>
#include <hash.h>
#include <util.h>
#include <cfg_struct.h>
#include <cfg_util.h>
#include <data_type.h>



struct cfg_struct_struct
{
  char * struct_name;
  char * name;

  hash_type * sub_items;   /* Hash indexed with items. All values are strings. */
  hash_type * sub_structs; /* Hash indexed with names of struct instances. All values are cfg_struct_types. */
};



static cfg_struct_type * cfg_struct_alloc_from_buffer(cfg_struct_def_type *, char **, char **, const char *, const char *, set_type *, bool);





/*
  NOTE: This is *not* the same prim_enum as in cfg_struct_def. The primitives are simply different
        when parsing a configuration spec and a given configuration.
*/

#define STR_CFG_ASSIGN      "="
#define STR_CFG_SCOPE_START "{"
#define STR_CFG_SCOPE_STOP  "}"
#define STR_CFG_END         ";"
#define STR_CFG_INCLUDE     "include"
#define STR_CFG_PAR_START   "("
#define STR_CFG_PAR_END     ")"



typedef enum {CFG_ASSIGN, CFG_SCOPE_START, CFG_SCOPE_STOP, CFG_END, CFG_INCLUDE, CFG_PAR_START, CFG_PAR_END, CFG_STRUCT, CFG_ITEM} prim_enum;



#define RETURN_PRIM_IF_MATCH_MACRO(PRIM, STRING) if(!strcmp(STRING, STR_## PRIM)){return PRIM;}
static prim_enum get_prim_from_str(cfg_struct_def_type * cfg_struct_def, const char * str)
{
  if(str == NULL)
    util_abort("%s: Trying to valdiate NULL as a token.\n");

  RETURN_PRIM_IF_MATCH_MACRO(CFG_ASSIGN, str);
  RETURN_PRIM_IF_MATCH_MACRO(CFG_SCOPE_START, str);
  RETURN_PRIM_IF_MATCH_MACRO(CFG_SCOPE_STOP, str);
  RETURN_PRIM_IF_MATCH_MACRO(CFG_END, str);
  RETURN_PRIM_IF_MATCH_MACRO(CFG_INCLUDE, str);
  RETURN_PRIM_IF_MATCH_MACRO(CFG_PAR_START, str);
  RETURN_PRIM_IF_MATCH_MACRO(CFG_PAR_END, str);

  if(hash_has_key(cfg_struct_def->sub_items, str))
    return CFG_ITEM;
  if(hash_has_key(cfg_struct_def->sub_structs, str))
    return CFG_STRUCT;

  util_abort("%s: \"%s\" is not a known primitive.\n", __func__, str);
  return 0;
}
#undef RETURN_PRIM_IF_MATCH_MACRO



#define RETURN_TRUE_IF_MATCH_MACRO(PRIM, STRING) if(!strcmp(STRING, STR_## PRIM)){return true;}
static bool str_is_prim(cfg_struct_def_type * cfg_struct_def, const char * str)
{
  if(str == NULL)
    util_abort("%s: Trying to valdiate NULL as a token.\n");

  RETURN_TRUE_IF_MATCH_MACRO(CFG_ASSIGN, str);
  RETURN_TRUE_IF_MATCH_MACRO(CFG_SCOPE_START, str);
  RETURN_TRUE_IF_MATCH_MACRO(CFG_SCOPE_STOP, str);
  RETURN_TRUE_IF_MATCH_MACRO(CFG_END, str);
  RETURN_TRUE_IF_MATCH_MACRO(CFG_INCLUDE, str);
  RETURN_TRUE_IF_MATCH_MACRO(CFG_PAR_START, str);
  RETURN_TRUE_IF_MATCH_MACRO(CFG_PAR_END, str);

  if(hash_has_key(cfg_struct_def->sub_items, str))
    return true;
  if(hash_has_key(cfg_struct_def->sub_structs, str))
    return true;

  return false;
}
#undef RETURN_TRUE_IF_MATCH_MACRO



static bool validate_token(cfg_struct_def_type * cfg_struct_def, prim_enum prim, const char * token)
{
  bool   ok = false;
  if(token == NULL)
    util_abort("%s: Expected a string, got NULL.\n", __func__);
  else if(!str_is_prim(cfg_struct_def, token))
    ok = false;
  else if(get_prim_from_str(cfg_struct_def, token) != prim)
    ok = false;
  else
    ok = true;
  return ok;
}




/**************************************************************************************************************************************/



void cfg_struct_free(cfg_struct_type * cfg_struct)
{
  free(cfg_struct->struct_name);
  free(cfg_struct->name);
  hash_free(cfg_struct->sub_items);
  hash_free(cfg_struct->sub_structs);
  free(cfg_struct);
}



void cfg_struct_free__(void * cfg_struct)
{
  cfg_struct_free( (cfg_struct_type *) cfg_struct);
}



/**************************************************************************************************************************************/



static void cfg_struct_set_item
(
          cfg_struct_def_type * cfg_struct_def,
          const char * item,
          char ** __buffer_pos,
          cfg_struct_type * cfg_struct
)
{
  cfg_item_def_type * cfg_item_def = hash_get(cfg_struct_def->sub_items, item);

  char * token = cfg_util_alloc_next_token(__buffer_pos);
  if(!validate_token(cfg_struct_def, CFG_ASSIGN, token))
    util_abort("%s: Syntax error. Expected \"%s\", got \"%s\".\n", __func__, STR_CFG_ASSIGN, token);

  free(token);
  token = cfg_util_alloc_next_token(__buffer_pos);
  if(validate_str_as_data_type(cfg_item_def->data_type, token))
    hash_insert_string(cfg_struct->sub_items, item, token);
  else
    util_abort("%s: Failed to parse \"%s\" to be of type \"%s\".\n", __func__, token, get_data_type_str_ref(cfg_item_def->data_type));

  free(token);
  token = cfg_util_alloc_next_token(__buffer_pos);
  if(!validate_token(cfg_struct_def, CFG_END, token))
    util_abort("%s: Syntax error. Expected \"%s\", got \"%s\".\n", __func__, STR_CFG_END, token);
  free(token);
}



static void cfg_struct_alloc_sub_struct
(
            cfg_struct_def_type * cfg_struct_def,
            char ** __buffer,
            char ** __buffer_pos,
            const char * struct_name,
            set_type * src_files,
            cfg_struct_type * cfg_struct
)
{
  char * name = cfg_util_alloc_next_token(__buffer_pos);
  if(str_is_prim(cfg_struct_def, name))
    util_abort("%s: Syntax error. Expected a string. Got primitive \"%s\".\n", __func__, name);
  if(hash_has_key(cfg_struct->sub_structs, name))
    util_abort("%s: Error, \"%s\" has already been defined.\n", __func__, name);

  cfg_struct_def_type * cfg_struct_def_sub = hash_get(cfg_struct_def->sub_structs, struct_name);
  cfg_struct_type * cfg_struct_sub = cfg_struct_alloc_from_buffer(cfg_struct_def_sub, __buffer, __buffer_pos, struct_name, name, src_files, false);

  hash_insert_hash_owned_ref(cfg_struct->sub_structs, name, cfg_struct_sub, cfg_struct_free__);
  free(name);
    
}




static cfg_struct_type * cfg_struct_alloc_from_buffer
(
                        cfg_struct_def_type * cfg_struct_def,
                        char ** __buffer,
                        char ** __buffer_pos,
                        const char * struct_name,
                        const char * name,
                        set_type * src_files,
                        bool is_root
)
{
  assert(name != NULL);

  cfg_struct_type * cfg_struct = util_malloc(sizeof * cfg_struct, __func__);
  
  cfg_struct->struct_name = util_alloc_string_copy(struct_name);
  cfg_struct->name        = util_alloc_string_copy(name);
  cfg_struct->sub_items   = hash_alloc();
  cfg_struct->sub_structs = hash_alloc();



  bool scope_start_set = false;
  bool scope_end_set   = false;
  bool struct_finished = false;

  for(;;)
  {
    char * token = cfg_util_alloc_next_token(__buffer_pos);

    /*
      First, check if something has gone haywire or if we are at end of buffer.
    */
    if(token == NULL && !is_root)
      util_abort("%s: Syntax error in struct \"%s\". Unexpected end of file.\n", __func__, name);
    else if(token == NULL && is_root)
    {
      struct_finished = true;
      break;
    }
    else if(!str_is_prim(cfg_struct_def, token))
      util_abort("%s: Syntax error in struct \"%s\". Expected primitive, got \"%s\".\n", __func__, name, token);
    
    prim_enum prim = get_prim_from_str(cfg_struct_def, token);
    switch(prim)
    {
      case(CFG_ASSIGN):
        util_abort("%s: Syntax error in struct \"%s\". Unexpected \"%s\".", __func__, name, STR_CFG_ASSIGN);
        break;
      case(CFG_SCOPE_START):
        if(scope_start_set)
          util_abort("%s: Syntax error in struct \"%s\". Unexpected \"%s\".", __func__, name, STR_CFG_SCOPE_START);
        else
          scope_start_set = true;
        break;
      case(CFG_SCOPE_STOP):
        if(!scope_start_set)
          util_abort("%s: Syntax error in struct \"%s\". Unexpected \"%s\".", __func__, name, STR_CFG_SCOPE_STOP);
        scope_end_set = true;
        struct_finished = true;
        break;
      case(CFG_END):
        util_abort("%s: Syntax error in struct \"%s\". Unexpected \"%s\".", __func__, name, STR_CFG_END);
        break;
      case(CFG_INCLUDE):
        util_abort("%s: Sorry, no support for include yet!.\n");
        break;
      case(CFG_PAR_START):
        util_abort("%s: Syntax error in struct \"%s\". Unexpected \"%s\".", __func__, name, STR_CFG_PAR_START);
        break;
      case(CFG_PAR_END):
        util_abort("%s: Syntax error in struct \"%s\". Unexpected \"%s\".", __func__, name, STR_CFG_PAR_END);
        break;
      case(CFG_ITEM):
        cfg_struct_set_item(cfg_struct_def, token, __buffer_pos, cfg_struct);
        break;
      case(CFG_STRUCT):
        cfg_struct_alloc_sub_struct(cfg_struct_def, __buffer, __buffer_pos, token, src_files, cfg_struct);
        break;
      default:
        util_abort("%s: Syntax error. Expected a primitive token. Got \"%s\".\n", __func__, token);
        break;
    }
    free(token);

    if(struct_finished)
    {
      char * buffer_pos = *__buffer_pos;
      token = cfg_util_alloc_next_token(&buffer_pos);
      if(token == NULL)
        break;
      if(validate_token(cfg_struct_def, CFG_END, token))
        *__buffer_pos = buffer_pos;
      free(token);
      break;
    }
  }
  if(scope_start_set != scope_end_set)
    util_abort("%s: Syntax error in struct \"%s\". Could not match delimiters.\n", __func__, name); 

  // TODO Create and call a "cfg_validate_struct" on cfg_struct.
  return cfg_struct;
}



cfg_struct_type * cfg_struct_alloc_from_file(const char * filename, cfg_struct_def_type * cfg_struct_def)
{
  char * pad_keys[] = {"{","}","=",";","(",")"};
  char * buffer = cfg_util_alloc_token_buffer(filename, "--", 6, (const char **) pad_keys);
  char * buffer_pos = buffer;

  set_type * src_files = set_alloc_empty();
  // TODO Should add absolute path.
  set_add_key(src_files, filename);

  cfg_struct_type * cfg_struct  = cfg_struct_alloc_from_buffer(cfg_struct_def, &buffer, &buffer_pos, "root", "root", src_files, true);
  free(buffer);
  return cfg_struct;
}
