#include <assert.h>
#include <string.h>
#include <cfg_struct_def.h>
#include <cfg_util.h>
#include <util.h>


static cfg_struct_def_type * cfg_struct_def_alloc_from_buffer(char **, const char *, bool);

/**************************************************************************************************************************************/


#define STR_CFG_ASSIGN      "="
#define STR_CFG_SCOPE_START "{"
#define STR_CFG_SCOPE_STOP  "}"
#define STR_CFG_END         ";"



#define STR_CFG_STRUCT      "struct"
#define STR_CFG_ITEM        "item"
#define STR_CFG_REQUIRED    "required"
#define STR_CFG_DEFAULT     "default"
#define STR_CFG_RESTRICTION "restrict"
#define STR_CFG_HELP        "help"
#define STR_CFG_DATA_TYPE   "data_type"



typedef enum {CFG_ASSIGN, CFG_SCOPE_START, CFG_SCOPE_STOP, CFG_END,
              CFG_STRUCT, CFG_ITEM, CFG_REQUIRED, CFG_DEFAULT, CFG_RESTRICTION, CFG_HELP, CFG_DATA_TYPE}
              prim_enum;



/**************************************************************************************************************************************/



#define RETURN_PRIM_IF_MATCH_MACRO(PRIM, STRING) if(!strcmp(STRING, STR_## PRIM)){return PRIM;}
static prim_enum get_prim_from_str(const char * str)
{
  if(str == NULL)
    util_abort("%s: Trying to valdiate NULL as a token.\n");
  
  RETURN_PRIM_IF_MATCH_MACRO(CFG_ASSIGN, str);
  RETURN_PRIM_IF_MATCH_MACRO(CFG_SCOPE_START, str);
  RETURN_PRIM_IF_MATCH_MACRO(CFG_SCOPE_STOP, str);
  RETURN_PRIM_IF_MATCH_MACRO(CFG_END, str);
  RETURN_PRIM_IF_MATCH_MACRO(CFG_STRUCT, str);
  RETURN_PRIM_IF_MATCH_MACRO(CFG_ITEM, str);
  RETURN_PRIM_IF_MATCH_MACRO(CFG_REQUIRED, str);
  RETURN_PRIM_IF_MATCH_MACRO(CFG_DEFAULT, str);
  RETURN_PRIM_IF_MATCH_MACRO(CFG_RESTRICTION, str);
  RETURN_PRIM_IF_MATCH_MACRO(CFG_HELP, str);
  RETURN_PRIM_IF_MATCH_MACRO(CFG_DATA_TYPE, str);

  util_abort("%s: \"%s\" is not a known primitive.\n", __func__, str);
  return 0;
}
#undef RETURN_PRIM_IF_MATCH_MACRO



#define RETURN_TRUE_IF_MATCH_MACRO(PRIM, STRING) if(!strcmp(STRING, STR_## PRIM)){return true;}
static bool str_is_prim(const char * str)
{
  if(str == NULL)
    util_abort("%s: Trying to valdiate NULL as a token.\n");

  RETURN_TRUE_IF_MATCH_MACRO(CFG_ASSIGN, str);
  RETURN_TRUE_IF_MATCH_MACRO(CFG_SCOPE_START, str);
  RETURN_TRUE_IF_MATCH_MACRO(CFG_SCOPE_STOP, str);
  RETURN_TRUE_IF_MATCH_MACRO(CFG_END, str);
  RETURN_TRUE_IF_MATCH_MACRO(CFG_STRUCT, str);
  RETURN_TRUE_IF_MATCH_MACRO(CFG_ITEM, str);
  RETURN_TRUE_IF_MATCH_MACRO(CFG_REQUIRED, str);
  RETURN_TRUE_IF_MATCH_MACRO(CFG_DEFAULT, str);
  RETURN_TRUE_IF_MATCH_MACRO(CFG_RESTRICTION, str);
  RETURN_TRUE_IF_MATCH_MACRO(CFG_HELP, str);
  RETURN_TRUE_IF_MATCH_MACRO(CFG_DATA_TYPE, str);

  return false; 
}
#undef RETURN_TRUE_IF_MATCH_MACRO



static bool validate_token(prim_enum prim, const char * token)
{
  bool   ok = false;
  if(token == NULL)
    util_abort("%s: Expected a string, got NULL.\n", __func__);
  else if(!str_is_prim(token))
    ok = false;
  else if(get_prim_from_str(token) != prim)
    ok = false;
  else
    ok = true;
  return ok;
}



/**************************************************************************************************************************************/



void cfg_item_def_free(cfg_item_def_type * cfg_item_def)
{
  free(cfg_item_def->name);
  set_free(cfg_item_def->restriction);
  util_safe_free(cfg_item_def->help);
  util_safe_free(cfg_item_def->default_value);
  free(cfg_item_def);
}



void cfg_item_def_free__(void * cfg_item_def)
{
  cfg_item_def_free( (cfg_item_def_type *) cfg_item_def);
}



void cfg_struct_def_free(cfg_struct_def_type * cfg_struct_def)
{
  free(cfg_struct_def->name);
  hash_free(cfg_struct_def->sub_items);
  hash_free(cfg_struct_def->sub_structs);
  set_free(cfg_struct_def->required_sub_items);
  set_free(cfg_struct_def->required_sub_structs);
  util_safe_free(cfg_struct_def->help);
  free(cfg_struct_def);
}



void cfg_struct_def_free__(void * cfg_struct_def)
{
  cfg_struct_def_free( (cfg_struct_def_type *) cfg_struct_def);
}



/**************************************************************************************************************************************/



static void cfg_item_def_set_data_type(cfg_item_def_type * cfg_item_def, char ** __buffer_pos)
{
  char * token = cfg_util_alloc_next_token(__buffer_pos);
  if(!validate_token(CFG_ASSIGN, token))
    util_abort("%s: Syntax error. Expected \"%s\", got \"%s\".\n", __func__, STR_CFG_ASSIGN, token);

  free(token);
  token = cfg_util_alloc_next_token(__buffer_pos);
  cfg_item_def->data_type = get_data_type_from_string(token);

  free(token);
  token = cfg_util_alloc_next_token(__buffer_pos);
  if(!validate_token(CFG_END, token))
    util_abort("%s: Syntax error. Expected \"%s\", got \"%s\".\n", __func__, STR_CFG_END, token);
  free(token);
}



static void cfg_item_def_set_default(cfg_item_def_type * cfg_item_def, char ** __buffer_pos)
{
  char * token = cfg_util_alloc_next_token(__buffer_pos);
  if(!validate_token(CFG_ASSIGN, token))
    util_abort("%s: Syntax error. Expected \"%s\", got \"%s\".\n", __func__, STR_CFG_ASSIGN, token);

  free(token);
  token = cfg_util_alloc_next_token(__buffer_pos);
  if(str_is_prim(token))
    util_abort("%s: Syntax error. Expected a string. Got primitive \"%s\".\n", __func__, token);
  else
    cfg_item_def->default_value = token;

  free(token);
  token = cfg_util_alloc_next_token(__buffer_pos);
  if(!validate_token(CFG_END, token))
    util_abort("%s: Syntax error. Expected \"%s\", got \"%s\".\n", __func__, STR_CFG_END, token);
  free(token);
}



static void cfg_item_def_set_help(cfg_item_def_type * cfg_item_def, char ** __buffer_pos)
{
  char * token = cfg_util_alloc_next_token(__buffer_pos);
  if(!validate_token(CFG_ASSIGN, token))
    util_abort("%s: Syntax error. Expected \"%s\", got \"%s\".\n", __func__, STR_CFG_ASSIGN, token);

  free(token);
  token = cfg_util_alloc_next_token(__buffer_pos);
  if(str_is_prim(token))
    util_abort("%s: Syntax error. Expected a string. Got primitive \"%s\".\n", __func__, token);
  else
    cfg_item_def->help = token;

  token = cfg_util_alloc_next_token(__buffer_pos);
  if(!validate_token(CFG_END, token))
    util_abort("%s: Syntax error. Expected \"%s\", got \"%s\".\n", __func__, STR_CFG_END, token);
  free(token);
}



static void cfg_item_def_set_restriction(cfg_item_def_type * cfg_item_def, char ** __buffer_pos)
{
  char * token = cfg_util_alloc_next_token(__buffer_pos);
  if(!validate_token(CFG_ASSIGN, token))
    util_abort("%s: Syntax error. Expected \"%s\", got \"%s\".\n", __func__, STR_CFG_ASSIGN, token);

  free(token);

  char * buffer_pos = *__buffer_pos;
  for(;;)
  {
    token = cfg_util_alloc_next_token(&buffer_pos);  
    if(str_is_prim(token))
    {
      break;
    }
    else
    {
      set_add_key(cfg_item_def->restriction, token);
      free(token);
    }
  }

  if(!validate_token(CFG_END, token))
     util_abort("%s: Syntax error. Expected \"%s\", got \"%s\".\n", __func__, STR_CFG_END, token);

  free(token);
  *__buffer_pos = buffer_pos;
}




static cfg_item_def_type * cfg_item_def_alloc_from_buffer(char ** __buffer_pos, const char * name)
{
  assert(name != NULL);

  cfg_item_def_type * cfg_item_def = util_malloc(sizeof * cfg_item_def, __func__);

  /* Alloc a default type. */
  cfg_item_def->name          = util_alloc_string_copy(name);
  cfg_item_def->restriction   = set_alloc_empty();
  cfg_item_def->data_type     = DATA_TYPE_STR;
  cfg_item_def->default_value = NULL;
  cfg_item_def->help          = NULL;

  char * buffer_pos = *__buffer_pos; 
  bool scope_start_set = false;
  bool scope_end_set   = false;

  /*
    In this loop, it is an error if token is not parseable to a primitive or NULL.
  */
  for(;;)
  {
    char * token = cfg_util_alloc_next_token(&buffer_pos);

    /*
      First, check if something has gone haywire or if we are at end of buffer.
    */
    if(token == NULL)
      util_abort("%s: Syntax error in item \"%s\". Unexpected end of file.\n", __func__, name);
    else if(!str_is_prim(token))
      util_abort("%s: Syntax error in item \"%s\". Expected primitive, got \"%s\".\n", __func__, name, token);


    /*
      Parse the primitive.
    */
    bool item_finished = false;
    prim_enum prim = get_prim_from_str(token);
    switch(prim)
    {
      case(CFG_ASSIGN):
        util_abort("%s: Syntax error in item \"%s\". Unexpected \"%s\".", __func__, name, STR_CFG_ASSIGN);
        break;
      case(CFG_SCOPE_START):
        if(scope_start_set)
          util_abort("%s: Syntax error in item \"%s\". Unexpected \"%s\".", __func__, name, STR_CFG_SCOPE_START);
        else
          scope_start_set = true;
        break;
      case(CFG_SCOPE_STOP):
        if(!scope_start_set)
          util_abort("%s: Syntax error in item \"%s\". Unexpected \"%s\".", __func__, name, STR_CFG_SCOPE_STOP);
        scope_end_set = true;
        item_finished = true;
        break;
      case(CFG_END):
        if(scope_start_set)
          util_abort("%s: Syntax error in item \"%s\". Unexpected \"%s\".", __func__, name, STR_CFG_END);
        item_finished = true;
        break;
      case(CFG_STRUCT):
        util_abort("%s: Syntax error in item \"%s\". Unexpected \"%s\".", __func__, name, STR_CFG_STRUCT);
        break;
      case(CFG_ITEM):
        util_abort("%s: Syntax error in item \"%s\". Unexpected \"%s\".", __func__, name, STR_CFG_ITEM);
        break;
      case(CFG_REQUIRED):
        util_abort("%s: Syntax error in item \"%s\". Unexpected \"%s\".", __func__, name, STR_CFG_REQUIRED);
        break;
      case(CFG_DEFAULT):
        cfg_item_def_set_default(cfg_item_def, &buffer_pos);
        break;
      case(CFG_RESTRICTION):
        if(!scope_start_set)
          util_abort("%s: Syntax error after item \"%s\". Unexpected \"%s\".", __func__, name, STR_CFG_RESTRICTION);
        else
          cfg_item_def_set_restriction(cfg_item_def, &buffer_pos);
        break;
      case(CFG_HELP):
        if(!scope_start_set)
          util_abort("%s: Syntax error after item \"%s\". Unexpected \"%s\".", __func__, name, STR_CFG_HELP);
        else
          cfg_item_def_set_help(cfg_item_def, &buffer_pos);
        break;
      case(CFG_DATA_TYPE):
        if(!scope_start_set)
          util_abort("%s: Syntax error after item \"%s\". Unexpected \"%s\".", __func__, name, STR_CFG_DATA_TYPE);
        else
          cfg_item_def_set_data_type(cfg_item_def, &buffer_pos);
        break;
      default:
        util_abort("%s: Internal error.\n", __func__);
    }
    free(token);


    if(item_finished)
    {
      *__buffer_pos = buffer_pos;
      if(prim == CFG_SCOPE_STOP)
      {
        token = cfg_util_alloc_next_token(&buffer_pos);
        if(validate_token(CFG_END, token))
          *__buffer_pos = buffer_pos;
        free(token);
      }
      break;
    }

  }
  return cfg_item_def;
}



static void cfg_struct_def_set_required(cfg_struct_def_type * cfg_struct_def, char ** __buffer_pos)
{
  bool   item = false;
  // For this token, we need to purely LOOK a head.
  char * buffer_pos = *__buffer_pos;
  char * token = cfg_util_alloc_next_token(&buffer_pos);
  if(!validate_token(CFG_ITEM, token) && !validate_token(CFG_STRUCT, token))
    util_abort("%s: Syntax error. Expected \"%s\" or \"%s\", got \"%s\".\n", __func__, STR_CFG_ITEM, STR_CFG_STRUCT , token);

  if(validate_token(CFG_ITEM, token))
    item = true;

  free(token);
  token = cfg_util_alloc_next_token(&buffer_pos);
  if(str_is_prim(token))
    util_abort("%s: Syntax error. Expected a string. Got primitive \"%s\".\n", __func__, token);
  else  
  {
    if(item)
      set_add_key(cfg_struct_def->required_sub_items, token);
    else
      set_add_key(cfg_struct_def->required_sub_structs, token);
  }

  free(token);
}



static void cfg_struct_def_alloc_sub_struct(cfg_struct_def_type * cfg_struct_def, char ** __buffer_pos)
{
  char * name_sub = cfg_util_alloc_next_token(__buffer_pos);
  if(str_is_prim(name_sub))
    util_abort("%s: Syntax error. Expected a string. Got primitive \"%s\".\n", __func__, name_sub);
  
  if(hash_has_key(cfg_struct_def->sub_items, name_sub))
    util_abort("%s: Syntax error. String \"%s\" has previously been used for an item in the same scope.\n", __func__, name_sub);
  if(hash_has_key(cfg_struct_def->sub_structs, name_sub))
    util_abort("%s: Syntax error. String \"%s\" has previously been used for a struct in the same scope.\n", __func__, name_sub);
    

  cfg_struct_def_type * cfg_struct_def_sub = cfg_struct_def_alloc_from_buffer(__buffer_pos, name_sub, false);
  hash_insert_hash_owned_ref(cfg_struct_def->sub_structs, name_sub, cfg_struct_def_sub, cfg_struct_def_free__);
  free(name_sub);
}



static void cfg_struct_def_alloc_sub_item(cfg_struct_def_type * cfg_struct_def, char ** __buffer_pos)
{
  char * name_sub = cfg_util_alloc_next_token(__buffer_pos);
  if(str_is_prim(name_sub))
    util_abort("%s: Syntax error. Expected a string. Got primitive \"%s\".\n", __func__, name_sub);
  
  if(hash_has_key(cfg_struct_def->sub_items, name_sub))
    util_abort("%s: Syntax error. String \"%s\" has previously been used for an item in the same scope.\n", __func__, name_sub);
  if(hash_has_key(cfg_struct_def->sub_structs, name_sub))
    util_abort("%s: Syntax error. String \"%s\" has previously been used for a struct in the same scope.\n", __func__, name_sub);

  cfg_item_def_type * cfg_item_def_sub = cfg_item_def_alloc_from_buffer(__buffer_pos, name_sub);
  hash_insert_hash_owned_ref(cfg_struct_def->sub_items, name_sub, cfg_item_def_sub, cfg_item_def_free__);
  free(name_sub);
}



static void cfg_struct_def_set_help(cfg_struct_def_type * cfg_struct_def, char ** __buffer_pos)
{
  char * token = cfg_util_alloc_next_token(__buffer_pos);
  if(!validate_token(CFG_ASSIGN, token))
    util_abort("%s: Syntax error. Expected \"%s\", got \"%s\".\n", __func__, STR_CFG_ASSIGN, token);

  free(token);
  token = cfg_util_alloc_next_token(__buffer_pos);
  if(str_is_prim(token))
    util_abort("%s: Syntax error. Expected a string. Got primitive \"%s\".\n", __func__, token);
  else
    cfg_struct_def->help = token;

  token = cfg_util_alloc_next_token(__buffer_pos);
  if(!validate_token(CFG_END, token))
    util_abort("%s: Syntax error. Expected \"%s\", got \"%s\".\n", __func__, STR_CFG_END, token);
  free(token);
}



static cfg_struct_def_type * cfg_struct_def_alloc_from_buffer(char ** __buffer_pos, const char * name, bool is_root)
{
  assert(name != NULL);

  cfg_struct_def_type * cfg_struct_def = util_malloc(sizeof * cfg_struct_def, __func__);

  cfg_struct_def->name                 = util_alloc_string_copy(name);
  cfg_struct_def->sub_items            = hash_alloc();
  cfg_struct_def->sub_structs          = hash_alloc();
  cfg_struct_def->required_sub_items   = set_alloc_empty();
  cfg_struct_def->required_sub_structs = set_alloc_empty();
  cfg_struct_def->help                 = NULL;



  char * buffer_pos = *__buffer_pos; 
  bool scope_start_set = false;
  bool scope_end_set   = false;
  bool struct_finished = false;
  bool struct_empty    = true;

  /*
    In this loop, it is an error if token is not parseable to a primitive or NULL.
  */
  for(;;)
  {
    char * token = cfg_util_alloc_next_token(&buffer_pos);

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
    else if(!str_is_prim(token))
      util_abort("%s: Syntax error in struct \"%s\". Expected primitive, got \"%s\".", __func__, name, token);

    prim_enum prim = get_prim_from_str(token);
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
      case(CFG_STRUCT):
        if(!scope_start_set && !is_root)
          util_abort("%s: Syntax error in struct \"%s\". Unexpected \"%s\".", __func__, name, STR_CFG_STRUCT);
        cfg_struct_def_alloc_sub_struct(cfg_struct_def, &buffer_pos);
        struct_empty = false;
        break;
      case(CFG_ITEM):
        if(!scope_start_set && !is_root)
          util_abort("%s: Syntax error in struct \"%s\". Unexpected \"%s\".", __func__, name, STR_CFG_ITEM);
        cfg_struct_def_alloc_sub_item(cfg_struct_def, &buffer_pos);
        struct_empty = false;
        break;
      case(CFG_REQUIRED):
        if(!scope_start_set && !is_root)
          util_abort("%s: Syntax error in struct \"%s\". Unexpected \"%s\".", __func__, name, STR_CFG_REQUIRED);
        cfg_struct_def_set_required(cfg_struct_def, &buffer_pos);
        break;
      case(CFG_DEFAULT):
        util_abort("%s: Syntax error in struct \"%s\". Unexpected \"%s\".", __func__, name, STR_CFG_DEFAULT);
        break;
      case(CFG_RESTRICTION):
      {
        util_abort("%s: Syntax error in struct \"%s\". Unexpected \"%s\".", __func__, name, STR_CFG_RESTRICTION);
        break;
      }
      case(CFG_HELP):
      {
        if(!scope_start_set && !is_root)
          util_abort("%s: Syntax error in struct \"%s\". Unexpected \"%s\".", __func__, name, STR_CFG_HELP);
        cfg_struct_def_set_help(cfg_struct_def, &buffer_pos);
        break;
      }
      case(CFG_DATA_TYPE):
      {
        util_abort("%s: Syntax error in struct \"%s\". Unexpected \"%s\".", __func__, name, STR_CFG_DATA_TYPE);
        break;
      }
      default:
        util_abort("%s: Internal error.\n", __func__);
    }
    free(token);
    *__buffer_pos = buffer_pos;

    if(struct_finished)
    {
      token = cfg_util_alloc_next_token(&buffer_pos);
      if(token == NULL)
        break;
      if(validate_token(CFG_END, token))
        *__buffer_pos = buffer_pos;
      free(token);
      break;
    }
  }
  if(scope_start_set != scope_end_set)
    util_abort("%s: Syntax error in struct \"%s\". Could not match delimiters.\n", __func__, name); 
  if(struct_empty)
    util_abort("%s: Syntax error. The struct \"%s\" is empty, this is not allowed.\n", __func__, name); 

  return cfg_struct_def;
}



/**************************************************************************************************************************************/



cfg_struct_def_type * cfg_struct_def_alloc_from_file(const char * filename)
{
  char * pad_keys[] = {"{","}","=",";"};
  char * buffer = cfg_util_alloc_token_buffer(filename, "--", 4, (const char **) pad_keys);
  char * buffer_pos = buffer;

  cfg_struct_def_type * cfg_struct_def  = cfg_struct_def_alloc_from_buffer(&buffer_pos, "root", true);
  free(buffer);
  return cfg_struct_def;
  return NULL;
}
