#include <assert.h>
#include <string.h>
#include <util.h>
#include <cfg_util.h>
#include <cfg_lang.h>


// Language primitivies
#define ASSIGMENT_STRING "="
#define SUB_START_STRING "{"
#define SUB_END_STRING   "}"



bool is_scope_end(const char * str)
{
  if(!strcmp(str, SUB_END_STRING)) return true;
  else                             return false;
}



bool is_scope_start(const char * str)
{
  if(!strcmp(str, SUB_START_STRING)) return true;
  else                               return false;
}



bool is_assignment(const char * str)
{
  if(!strcmp(str, ASSIGMENT_STRING)) return true;
  else                               return false;
}


/**************************************************************************************/
#define DATA_TYPE_STR_STRING      "string"
#define DATA_TYPE_INT_STRING      "int"
#define DATA_TYPE_POSINT_STRING   "posint"
#define DATA_TYPE_FLOAT_STRING    "float"
#define DATA_TYPE_POSFLOAT_STRING "posfloat"
#define DATA_TYPE_FILE_STRING     "file"
#define DATA_TYPE_DATE_STRING     "date"



typedef enum {TYPE, KEY, DATA_TYPE, REQUIRED, RESTRICTION, HELP} key_enum;
#define TYPE_STRING        "type"
#define KEY_STRING         "key"
#define DATA_TYPE_STRING   "data_type"
#define REQUIRED_STRING    "required"
#define RESTRICTION_STRING "restriction"
#define HELP_STRING        "help_text"



#define RETURN_TYPE_IF_MATCH(STRING,TYPE) if(strcmp(STRING, TYPE ##_STRING) == 0){ return TYPE;}
static key_enum get_key_from_string(const char * str)
{
  RETURN_TYPE_IF_MATCH(str, TYPE);
  RETURN_TYPE_IF_MATCH(str, KEY);
  RETURN_TYPE_IF_MATCH(str, DATA_TYPE);
  RETURN_TYPE_IF_MATCH(str, REQUIRED);
  RETURN_TYPE_IF_MATCH(str, RESTRICTION);
  RETURN_TYPE_IF_MATCH(str, HELP);

  util_abort("%s: Key \"%s\" is unkown.\n", __func__, str);
  return 0;
}
#undef RETURN_TYPE_IF_MATCH



static bool is_key_key(const char * str)
{
  if(!strcmp(str, KEY_STRING)) return true;
  else                         return false;
}


static bool is_key_type(const char * str)
{
  if(     !strcmp(str, TYPE_STRING)) return true;
  else                               return false;
}


static bool is_key(const char * str)
{
  if(     !strcmp(str, TYPE_STRING       )) return true;
  else if(!strcmp(str, KEY_STRING        )) return true;
  else if(!strcmp(str, DATA_TYPE_STRING  )) return true;
  else if(!strcmp(str, REQUIRED_STRING   )) return true;
  else if(!strcmp(str, RESTRICTION_STRING)) return true;
  else if(!strcmp(str, HELP_STRING       )) return true;
  else                                      return false;
}


static bool is_language(const char * str)
{
  if(     is_key(        str)) return true;
  else if(is_data_type(  str)) return true;
  else if(is_scope_start(str)) return true;
  else if(is_scope_end(  str)) return true;
  else if(is_assignment( str)) return true;
  else                         return false;

}



void cfg_key_def_free(cfg_key_def_type * cfg_key_def)
{
  free(cfg_key_def->name);
  set_free(cfg_key_def->restriction_set);
  util_safe_free(cfg_key_def->help_text);
}



void cfg_key_def_free__(void * cfg_key_def)
{
  cfg_key_def_free( (cfg_key_def_type *) cfg_key_def);
}



void cfg_type_def_free(cfg_type_def_type * cfg_type_def)
{
  free(cfg_type_def->name);
  hash_free(cfg_type_def->sub_keys);
  hash_free(cfg_type_def->sub_types);
  set_free(cfg_type_def->required);
  util_safe_free(cfg_type_def->help_text);
}



void cfg_type_def_free__(void * cfg_type_def)
{
  cfg_type_def_free( (cfg_type_def_type *) cfg_type_def);
}



void cfg_key_def_printf(const cfg_key_def_type * cfg_key_def)
{
  printf("Summary of key \"%s\":\n",cfg_key_def->name );
  printf("Help text:\n%s\n", cfg_key_def->help_text);

  int num_restrictions = set_get_size(cfg_key_def->restriction_set); 
  char ** restrictions = set_alloc_keylist(cfg_key_def->restriction_set);

  if(num_restrictions > 0)
  {
    printf("allowed values:\n-------------------\n");
    for(int i=0; i<num_restrictions; i++)
      printf("%i : %s\n", i, restrictions[i]);
  }
  util_free_stringlist(restrictions, num_restrictions);
}



void cfg_type_def_printf(const cfg_type_def_type * cfg_type_def)
{
  printf("Summary of type \"%s\":\n", cfg_type_def->name);
  printf("Help text:\n%s\n\n", cfg_type_def->help_text);

  int num_sub_keys = hash_get_size(cfg_type_def->sub_keys);
  char ** sub_keys = hash_alloc_keylist(cfg_type_def->sub_keys);
  if(num_sub_keys > 0)
  {
    printf("Defined sub keys:\n------------------\n");
    for(int i=0; i<num_sub_keys; i++)
      printf("%i : %s\n", i, sub_keys[i]);
  }
  printf("\n");
  util_free_stringlist(sub_keys, num_sub_keys);

  int num_sub_types = hash_get_size(cfg_type_def->sub_types);
  char ** sub_types = hash_alloc_keylist(cfg_type_def->sub_types);
  if(num_sub_types > 0)
  {
    printf("Defined sub types:\n------------------\n");
    for(int i=0; i<num_sub_types; i++)
      printf("%i : %s\n", i, sub_types[i]);
  }
  printf("\n");
  util_free_stringlist(sub_types, num_sub_types);

  int num_required = set_get_size(cfg_type_def->required);
  char ** required = set_alloc_keylist(cfg_type_def->required);
  if(num_required > 0)
  {
    printf("Required sub keys and types:\n------------------------\n");
    for(int i=0; i<num_required; i++)
      printf("%i : %s\n", i, required[i]);
  }
  printf("\n");
}



static void cfg_key_def_set_data_type_from_buffer(cfg_key_def_type * cfg_key_def, char ** __buffer_pos)
{
  char * asgnt = cfg_util_alloc_next_token(__buffer_pos);
  if(asgnt == NULL)
    util_abort("%s: Syntax error in key definition \"%s\". Expected \"%s\" after \"%s\", got NULL.\n", __func__, cfg_key_def->name, ASSIGMENT_STRING, DATA_TYPE_STRING);
  else if(!is_assignment(asgnt))
    util_abort("%s: Syntax error in key definition \"%s\". Expected \"%s\" after \"%s\", got \"%s\".\n", __func__, cfg_key_def->name, ASSIGMENT_STRING, DATA_TYPE_STRING, asgnt);
  else
    free(asgnt);

  char * token = cfg_util_alloc_next_token(__buffer_pos);

  if(!is_data_type(token))
    util_abort("%s: Syntax error in key definition \"%s\". Expected a data_type, got %s.\n", __func__, cfg_key_def->name, token);

  cfg_key_def->data_type = get_data_type_from_string(token);
  free(token);
}



static void cfg_key_def_set_restriction_from_buffer(cfg_key_def_type * cfg_key_def, char ** __buffer_pos)
{
  char * asgnt = cfg_util_alloc_next_token(__buffer_pos);
  if(asgnt == NULL)
    util_abort("%s: Syntax error in key definition \"%s\". Expected \"%s\" after \"%s\", got NULL.\n", __func__, cfg_key_def->name, ASSIGMENT_STRING, RESTRICTION_STRING);
  else if(!is_assignment(asgnt))
    util_abort("%s: Syntax error in key definition \"%s\". Expected \"%s\" after \"%s\", got \"%s\".\n", __func__, cfg_key_def->name, ASSIGMENT_STRING, RESTRICTION_STRING, asgnt);
  else
    free(asgnt);

  char * __buffer_pos_wrk = *__buffer_pos;
  for(;;)
  {
    char * token = cfg_util_alloc_next_token(&__buffer_pos_wrk);  
    if(token == NULL)
    {
      break;
    }
    else if(is_language(token))
    {
      free(token);
      break;
    }
    else
    {
      printf("adding restriction \"%s\" to key \"%s\".\n", token, cfg_key_def->name);
      set_add_key(cfg_key_def->restriction_set, token);
      free(token);
      *__buffer_pos = __buffer_pos_wrk;
    }
  }
}



static void cfg_key_def_set_help_text_from_buffer(cfg_key_def_type * cfg_key_def, char ** __buffer_pos)
{
  char * asgnt = cfg_util_alloc_next_token(__buffer_pos);
  if(asgnt == NULL)
    util_abort("%s: Syntax error in key definition \"%s\". Expected \"%s\" after \"%s\", got NULL.\n", __func__, cfg_key_def->name, ASSIGMENT_STRING, HELP_STRING);
  else if(!is_assignment(asgnt))
    util_abort("%s: Syntax error in key definition \"%s\". Expected \"%s\" after \"%s\", got \"%s\".\n", __func__, cfg_key_def->name, ASSIGMENT_STRING, HELP_STRING, asgnt);
  else
    free(asgnt);

  char * token = cfg_util_alloc_next_token(__buffer_pos);
  if(token == NULL)
    util_abort("%s: Syntax error in key definition \"%s\". Expected argument to \"%s\", got NULL.\n", __func__, cfg_key_def->name, HELP_STRING);
  else if(is_language(token))
    util_abort("%s: Syntax error in key definition \"%s\". Expected argument to \"%s\", got language key \"%s\".\n", __func__, cfg_key_def->name, HELP_STRING, token);
  else
    cfg_key_def->help_text = token;
}



cfg_key_def_type * cfg_key_def_alloc_from_buffer(char ** __buffer, char ** __buffer_pos, const char * name)
{
  assert(name != NULL);

  cfg_key_def_type * cfg_key_def = util_malloc(sizeof * cfg_key_def, __func__);

  /* Defaults. */
  cfg_key_def->name            = util_alloc_string_copy(name);
  cfg_key_def->restriction_set = set_alloc_empty();
  cfg_key_def->data_type       = DATA_TYPE_STR;
  cfg_key_def->help_text       = NULL;

  /* Current position in the buffer. */
  char * buffer_pos = *__buffer_pos;
  
  bool scope_start_set = false;
  for(;;)
  {
    char * token = cfg_util_alloc_next_token(&buffer_pos);


    if(token == NULL)
    {
      if(scope_start_set)
      {
        util_abort("%s: Syntax error, could not match delimiters in key definition \"%s\".", __func__, name);
      }
      else
      {
        free(token);
        break;
      }
    }
    else if(is_scope_start(token))
    {
      scope_start_set = true;
      continue;
    }
    else if(is_scope_end(token))
    {
      if(scope_start_set)
      {
        // We are done, move the calling scope's pos past the scope end.
        *__buffer_pos = buffer_pos;
        free(token);
        break;
      }
      else
      {
        // We are done, but have read the calling scope's "}". Dont move the pos.
        free(token);
        break;
      }
    }
    else if(is_assignment(token))
    {
      util_abort("%s: Syntax error in keyword %s, unexpected \"%s\".\n", __func__, name, ASSIGMENT_STRING);
    }
    else if(is_key(token))
    {
      bool end_of_key = false;
      key_enum key = get_key_from_string(token);
      switch(key)
      {
        case(TYPE):
        {
          if(scope_start_set)
            util_abort("%s: Syntax error in keyword %s, type definition not allowed inside key definition!\n", __func__, name);
          else
            end_of_key = true;
          break;
        }
        case(KEY):
          if(scope_start_set)
            util_abort("%s: Syntax error in keyword %s, recursive key definition is not allowed.\n", __func__, name);
          else
            end_of_key = true;
          break;
        case(DATA_TYPE):
          cfg_key_def_set_data_type_from_buffer(cfg_key_def, &buffer_pos);
          break;
        case(RESTRICTION):
          cfg_key_def_set_restriction_from_buffer(cfg_key_def, &buffer_pos );
          break;
        case(REQUIRED):
          if(scope_start_set)
            util_abort("%s: Syntax error in keyword %s, use of keyword %s is not allowed inside key definition!\n", __func__, name, REQUIRED_STRING);
          else
            end_of_key = true;
          break;
        case(HELP):
          cfg_key_def_set_help_text_from_buffer(cfg_key_def, &buffer_pos);
          break;
        default:
          util_abort("%s: Internal error.\n", __func__);
      }
      if(end_of_key)
      {
        free(token);
        break;
      }
    }
    else
    {
      if(scope_start_set && is_language(token))
      {
        util_abort("%s: Syntax error, could not match delimiters in key definition \"%s\".", __func__, name);
      }
      else if(!is_language(token))
      {
        util_abort("%s: Syntax error, unexpected expression \"%s\" in key definition \"%s\".", __func__, token, name);
      }
      else
      {
        free(token);
        break;
      }
    }

    free(token);
    *__buffer_pos = buffer_pos;
  }
  return cfg_key_def;
}



static void cfg_type_def_set_help_text_from_buffer(cfg_type_def_type * cfg_type_def, char ** __buffer_pos)
{
  char * asgnt = cfg_util_alloc_next_token(__buffer_pos);
  if(asgnt == NULL)
    util_abort("%s: Syntax error in type definition \"%s\". Expected \"%s\" after \"%s\", got NULL.\n", __func__, cfg_type_def->name, ASSIGMENT_STRING, HELP_STRING);
  else if(!is_assignment(asgnt))
    util_abort("%s: Syntax error in type definition \"%s\". Expected \"%s\" after \"%s\", got \"%s\".\n", __func__, cfg_type_def->name, ASSIGMENT_STRING, HELP_STRING, asgnt);
  else
    free(asgnt);

  char * token = cfg_util_alloc_next_token(__buffer_pos);
  if(token == NULL)
  {
    util_abort("%s: Syntax error in type definition \"%s\". Expected argument to \"%s\", got NULL.\n", __func__, cfg_type_def->name, HELP_STRING);
  }
  else if(is_language(token))
  {
    util_abort("%s: Syntax error in type definition \"%s\". Expected argument to \"%s\", got language type \"%s\".\n", __func__, cfg_type_def->name, HELP_STRING, token);
  }
  else
  {
    cfg_type_def->help_text = token;
  }
}



static void cfg_type_def_set_required_from_buffer(cfg_type_def_type * cfg_type_def, char ** __buffer_pos)
{
  // We need to look two steps ahead for this kw.
  char * buffer_pos_fwd = *__buffer_pos;
  char * sub_class = cfg_util_alloc_next_token(&buffer_pos_fwd);
  char * sub_name  = cfg_util_alloc_next_token(&buffer_pos_fwd);
 
  if(sub_class == NULL || sub_name == NULL)
  {
    util_abort("%s: Syntax error in type definition \"%s\". Keyword \"%s\" must be follwed by key or type and a name.\n", __func__, cfg_type_def->name, REQUIRED_STRING);
  }
  else if(!is_key_key(sub_class) && !is_key_type(sub_class))
  {
    util_abort("%s: Syntax error in type definition \"%s\". Keyword \"%s\" must be follwed by key or type.\n", __func__, cfg_type_def->name, REQUIRED_STRING);
  }
  else if(is_language(sub_name))
  {
    util_abort("%s: Syntax error in type definition \"%s\". Expected an identifier, got language \"%s\".\n", __func__, cfg_type_def->name, sub_name);
  }
  else
  {
    set_add_key(cfg_type_def->required, sub_name);
    free(sub_class);
    free(sub_name);
  }
}



cfg_type_def_type * cfg_type_def_alloc_from_buffer(char ** __buffer, char ** __buffer_pos, const char * name)
{
  assert(name != NULL);

  cfg_type_def_type * cfg_type_def = util_malloc(sizeof * cfg_type_def, __func__);

  cfg_type_def->name      = util_alloc_string_copy(name);
  cfg_type_def->sub_keys  = hash_alloc();
  cfg_type_def->sub_types = hash_alloc();
  cfg_type_def->required  = set_alloc_empty();
  cfg_type_def->help_text = NULL;

  /* Current position in the buffer. */
  char * buffer_pos = *__buffer_pos;

  bool scope_start_set = false;
  bool empty_type      = true;
  for(;;)
  {
    char * token = cfg_util_alloc_next_token(&buffer_pos);


    if(token == NULL)
    {
      if(scope_start_set)
      {
        util_abort("%s: Unexpected end of type definition \"%s\". Could not find a matching \"%s\".\n", __func__, name, SUB_END_STRING);
      }
      else if(empty_type)
      {
        util_abort("%s: Unexpected end of type definition \"%s\". Empty types are not allowed.\n", __func__, name);
      }
      else
      {
        free(token);
        break;
      }
    }
    else if(is_scope_start(token))
    {
      if(scope_start_set)
      {
        util_abort("%s: Syntax error, unexpected \"%s\" in type definition \"%s\".\n", __func__, SUB_START_STRING, name);
      }
      else
      {
        scope_start_set = true;
      }
    }
    else if(is_scope_end(token))
    {
      if(scope_start_set && !empty_type)
      {
        free(token);
        break;
      }
      else if(scope_start_set && empty_type)
      {
        util_abort("%s: Syntax error, unexpected \"%s\" in type definition \"%s\". Empty types are not allowed.\n", __func__, SUB_END_STRING, name);
      }
      else
      {
        util_abort("%s: Syntax error, unexpected \"%s\" in type definition \"%s\".\n", __func__, SUB_END_STRING, name);
      }
    }
    else if(is_key(token))
    {
      key_enum key = get_key_from_string(token);
      empty_type = false;
      switch(key)
      {
        case(TYPE):
        {
          char * sub_name = cfg_util_alloc_next_token(&buffer_pos);
          if(hash_has_key(cfg_type_def->sub_keys, sub_name))
            util_abort("%s: The name \"%s\" has previously been used for a key in the same scope.\n", __func__, sub_name);

          cfg_type_def_type * sub_cfg_type_def = cfg_type_def_alloc_from_buffer(__buffer, &buffer_pos, sub_name);
          hash_insert_hash_owned_ref(cfg_type_def->sub_types, sub_name, sub_cfg_type_def, cfg_type_def_free__);
          free(sub_name);
          break;
        }
        case(KEY):
        {
          char * sub_name = cfg_util_alloc_next_token(&buffer_pos);
          if(hash_has_key(cfg_type_def->sub_types, sub_name))
            util_abort("%s: The name \"%s\" has previously been used for a type in the same scope.\n", __func__, sub_name);

          cfg_key_def_type * sub_cfg_key_def = cfg_key_def_alloc_from_buffer(__buffer, &buffer_pos, sub_name);
          hash_insert_hash_owned_ref(cfg_type_def->sub_keys, sub_name, sub_cfg_key_def, cfg_key_def_free__);
          free(sub_name);
          break;
        }
        case(DATA_TYPE):
          util_abort("%s: Syntax error in type definition \"%s\", use of \"%s\" is not allowed in type definition.\n", __func__, name, DATA_TYPE_STRING);
          break;
        case(RESTRICTION):
          util_abort("%s: Syntax error in type definition \"%s\", use of \"%s\" is not allowed in type definition.\n", __func__, name, RESTRICTION_STRING);
          break;
        case(REQUIRED):
          cfg_type_def_set_required_from_buffer(cfg_type_def, &buffer_pos);
          break;
        case(HELP):
          cfg_type_def_set_help_text_from_buffer(cfg_type_def, &buffer_pos);
          break;
        default:
          util_abort("%s: Internal error.\n", __func__);
      }
    }
    else{
      util_abort("%s: Syntax error in type definition \"%s\". Expected key or scope, got \"%s\".\n", __func__, name, token);
    }
    free(token);
  }

  *__buffer_pos = buffer_pos;
  return cfg_type_def;
}



bool has_subtype(cfg_type_def_type * cfg_type_def, const char * str)
{
  if(hash_has_key(cfg_type_def->sub_types, str))
    return true;
  else
    return false;
}



bool has_subkey(cfg_type_def_type * cfg_type_def, const char * str)
{
  if(hash_has_key(cfg_type_def->sub_keys, str))
    return true;
  else
    return false;
}
