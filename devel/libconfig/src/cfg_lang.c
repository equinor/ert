#include <assert.h>
#include <string.h>
#include <hash.h>
#include <set.h>
#include <util.h>
#include <cfg_util.h>
#include <cfg_lang.h>


typedef enum {TYPE, KEY, DATA_TYPE, REQUIRED, RESTRICTION, HELP} key_enum;
// These are the primitive keys which are allowed in the definition.
#define TYPE_STRING        "type"
#define KEY_STRING         "key"
#define DATA_TYPE_STRING   "data_type"
#define REQUIRED_STRING    "required"
#define RESTRICTION_STRING "restriction"
#define HELP_STRING        "help_text"

typedef enum {SUB_START, SUB_END} scope_enum;
#define SUB_START_STRING "{"
#define SUB_END_STRING   "}"


typedef enum {DATA_TYPE_STR, DATA_TYPE_INT, DATA_TYPE_POSINT, DATA_TYPE_FLOAT, DATA_TYPE_POSFLOAT, DATA_TYPE_FILE, DATA_TYPE_DATE} data_type_enum;
#define DATA_TYPE_STR_STRING      "string"
#define DATA_TYPE_INT_STRING      "int"
#define DATA_TYPE_POSINT_STRING   "posint"
#define DATA_TYPE_FLOAT_STRING    "float"
#define DATA_TYPE_POSFLOAT_STRING "posfloat"
#define DATA_TYPE_FILE_STRING     "file"
#define DATA_TYPE_DATE_STRING     "date"



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



static scope_enum get_scope_from_string(const char * str)
{
  RETURN_TYPE_IF_MATCH(str, SUB_START);
  RETURN_TYPE_IF_MATCH(str, SUB_END);

  util_abort("%s: Scope modifier \"%s\" is unkown.\n", __func__, str);
  return 0;
}



static data_type_enum get_data_type_from_string(const char * str)
{
  RETURN_TYPE_IF_MATCH(str, DATA_TYPE_STR);
  RETURN_TYPE_IF_MATCH(str, DATA_TYPE_INT);
  RETURN_TYPE_IF_MATCH(str, DATA_TYPE_POSINT);
  RETURN_TYPE_IF_MATCH(str, DATA_TYPE_FLOAT);
  RETURN_TYPE_IF_MATCH(str, DATA_TYPE_POSFLOAT);
  RETURN_TYPE_IF_MATCH(str, DATA_TYPE_FILE);
  RETURN_TYPE_IF_MATCH(str, DATA_TYPE_DATE);

  util_abort("%s: Data type \"%s\" is unkown.\n", __func__, str);
  return 0;
}
#undef RETURN_TYPE_IF_MATCH



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



static bool is_scope(const char * str)
{
  if(     !strcmp(str, SUB_START_STRING)) return true;
  else if(!strcmp(str, SUB_END_STRING  )) return true;
  else                                    return false;
}



static bool is_scope_end(const char * str)
{
  if(!strcmp(str, SUB_END_STRING  )) return true;
  else                               return false;
}



static bool is_scope_start(const char * str)
{
  if(!strcmp(str, SUB_START_STRING  )) return true;
  else                               return false;
}



static bool is_data_type(const char * str)
{
  if(     !strcmp(str, DATA_TYPE_STR_STRING     )) return true;
  else if(!strcmp(str, DATA_TYPE_INT_STRING     )) return true;
  else if(!strcmp(str, DATA_TYPE_POSINT_STRING  )) return true;
  else if(!strcmp(str, DATA_TYPE_FLOAT_STRING   )) return true;
  else if(!strcmp(str, DATA_TYPE_POSFLOAT_STRING)) return true;
  else if(!strcmp(str, DATA_TYPE_FILE_STRING    )) return true;
  else if(!strcmp(str, DATA_TYPE_DATE_STRING    )) return true;
  else                                             return false;
}


static bool is_language(const char * str)
{
  if(     is_key(      str)) return true;
  else if(is_data_type(str)) return true;
  else if(is_scope(    str)) return true;
  else                       return false;

}



struct cfg_key_def_struct{
  char * name;
  data_type_enum   data_type;
  set_type       * restriction_set;

  char * help_text;
};


struct cfg_type_def_struct{
  char * name;

  hash_type  * sub_keys;  /* cfg_key_def_struct's. */
  hash_type  * sub_types; /* cfg_type_def_struct's. */

  set_type  * required; /* A list of key's and types that are required. */

  char * help_text;
};



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
  printf("Help text:\n%s\n", cfg_type_def->help_text);

  int num_sub_keys = hash_get_size(cfg_type_def->sub_keys);
  char ** sub_keys = hash_alloc_keylist(cfg_type_def->sub_keys);
  if(num_sub_keys > 0)
  {
    printf("Defined sub keys:\n------------------\n");
    for(int i=0; i<num_sub_keys; i++)
      printf("%i : %s\n", i, sub_keys[i]);
  }
  util_free_stringlist(sub_keys, num_sub_keys);

  int num_sub_types = hash_get_size(cfg_type_def->sub_types);
  char ** sub_types = hash_alloc_keylist(cfg_type_def->sub_types);
  if(num_sub_types > 0)
  {
    printf("Defined sub types:\n------------------\n");
    for(int i=0; i<num_sub_types; i++)
      printf("%i : %s\n", i, sub_types[i]);
  }
  util_free_stringlist(sub_types, num_sub_types);

  int num_required = set_get_size(cfg_type_def->required);
  char ** required = set_alloc_keylist(cfg_type_def->required);
  if(num_required > 0)
  {
    printf("Required sub keys and types:\n------------------------\n");
    for(int i=0; i<num_required; i++)
      printf("%i : %s\n", i, required[i]);


  }
}



static void cfg_key_def_set_data_type_from_buffer(cfg_key_def_type * cfg_key_def, char ** __buffer)
{

}



static void cfg_key_def_set_restriction_from_buffer(cfg_key_def_type * cfg_key_def, char ** __buffer)
{

}



static void cfg_key_def_set_help_text_from_buffer(cfg_key_def_type * cfg_key_def, char ** __buffer)
{

}



cfg_key_def_type * cfg_key_def_alloc_from_buffer(char ** __buffer_pos, const char * name)
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
  

  for(;;)
  {
    char * token = cfg_util_alloc_next_token(&buffer_pos);
    assert(token != NULL);

    if(is_scope_end(token))
    {
      free(token);
      break;
    }

    if(!is_key(token))
      util_abort("%s: Syntax error, expected language identifier. Got \"%s\".\n", __func__, token);

    key_enum key = get_key_from_string(token);
    switch(key)
    {
      case(TYPE):
        util_abort("%s: Syntax error, type definition not allowed inside key definition!\n", __func__);
        break;
      case(KEY):
        util_abort("%s: Syntax error, recursive key definition is not allowed.\n", __func__);
        break;
      case(DATA_TYPE):
        cfg_key_def_set_data_type_from_buffer(cfg_key_def, &buffer_pos);
        break;
      case(RESTRICTION):
        cfg_key_def_set_restriction_from_buffer(cfg_key_def, &buffer_pos );
        break;
      case(REQUIRED):
        util_abort("%s: Syntax error, use of keyword %s is not allowed inside key definition!\n", __func__, REQUIRED_STRING);
        break;
      case(HELP):
        cfg_key_def_set_help_text_from_buffer(cfg_key_def, &buffer_pos);
        break;
      default:
        util_abort("%s: Internal error.\n", __func__);
    }
    free(token);
  }
  __buffer_pos = &buffer_pos;
  return cfg_key_def;
}



cfg_type_def_type * cfg_type_def_alloc_from_tokens(int num_tokens, const char ** tokens, const char * name)
{
  assert(name != NULL);

  cfg_type_def_type * cfg_type_def = util_malloc(sizeof * cfg_type_def, __func__);

  cfg_type_def->name      = util_alloc_string_copy(name);
  cfg_type_def->sub_keys  = hash_alloc();
  cfg_type_def->sub_types = hash_alloc();
  cfg_type_def->required  = set_alloc_empty();
  cfg_type_def->help_text = NULL;

  int token_nr = 0;
  while(token_nr < num_tokens)
  {
    const char * token = tokens[token_nr];
    if(!is_key(token))
    {
      util_abort("%s: Syntax error, expected language key. Got \"%s\".\n", __func__, token);
    }

    key_enum key = get_key_from_string(token);

    switch(key)
    {
      case(REQUIRED):
      {
        // Assert that the next token is KEY or TYPE and that the following is not language. Add the following token to required.
        assert(token_nr + 2 < num_tokens);
        int token_nr_cpy = token_nr;
        token_nr_cpy++;
        token = tokens[token_nr_cpy];
        if(is_key(token))
        {
          key_enum next_key = get_key_from_string(token);
          if(next_key == TYPE || next_key == KEY)
          {
            token_nr_cpy++;
            token = tokens[token_nr_cpy];
            if(is_language(token))
            {
              util_abort("%s: Syntax error expected identifier, got \"%s\" which is language.\n", __func__, token);
            }
            else
            {
              set_add_key(cfg_type_def->required, token);
            }
          }
          else
          {
            util_abort("%s: Syntax error, \"%s\" must be folowed by \"%s\" or \"%s\"\n.", __func__, REQUIRED_STRING, TYPE_STRING, KEY_STRING);
          }
        }
        else
        {
          util_abort("%s: Syntax error, \"%s\" must be folowed by \"%s\" or \"%s\"\n.", __func__, REQUIRED_STRING, TYPE_STRING, KEY_STRING);
        }
        break;
      }
      case(TYPE):
      {
        // Assert that the next token is not language. Assert that the following token is SUB_START. Find the depth and alloc.
        // Add to sub_types. Move token_nr to SUB_END.
        assert(token_nr + 3 < num_tokens);
        token_nr++;
        token = tokens[token_nr];
        if(is_language(token))
        {
          util_abort("%s: Syntax error, expected identifier, got \"%s\" which is language.\n", __func__, token);
        }
        const char * name = token;
        token_nr++;
        token = tokens[token_nr];
        if(is_scope(token))
        {
          scope_enum scope = get_scope_from_string(token);
          if(scope == SUB_START)
          {
            int sub_size       = cfg_util_sub_size(num_tokens-token_nr, tokens+token_nr, SUB_START_STRING, SUB_END_STRING);
            int sub_size_strip = sub_size - 2;
            int offset         = token_nr + 1;

            if(sub_size_strip < 0)
              sub_size_strip = 0;

            cfg_type_def_type * cfg_type_def_sub = cfg_type_def_alloc_from_tokens(sub_size_strip, tokens+offset, name);
            hash_insert_hash_owned_ref(cfg_type_def->sub_types, name, cfg_type_def_sub, cfg_type_def_free__);

            token_nr = token_nr + sub_size - 1;
            token = tokens[token_nr];

            scope = get_scope_from_string(token);
            if(scope != SUB_END)
            {
              util_abort("%s: Syntax error, expected \"%s\" got \"%s\".\n", __func__, SUB_END_STRING, token);
            }
          }
          else
          {
            util_abort("%s: Syntax error, expected \"%s\" got \"%s\".\n", __func__, SUB_START_STRING, token);
          }
        }
        else
        {
          util_abort("%s: Syntax error, expected \"%s\" got \"%s\".\n", __func__, SUB_START_STRING, token);
        }
        break;
      }
      case(KEY):
      {
        // Assert that the next token is not language. Assert that the following token is SUB_START. Find the depth and alloc.
        // Add to sub_keys. Move token_nr past SUB_END.
        assert(token_nr + 3 < num_tokens);
        token_nr++;
        token = tokens[token_nr];
        if(is_language(token))
        {
          util_abort("%s: Syntax error, expected identifier, got \"%s\" which is language.\n", __func__, token);
        }
        const char * name = token;
        token_nr++;
        token = tokens[token_nr];
        if(is_scope(token))
        {
          scope_enum scope = get_scope_from_string(token);
          if(scope == SUB_START)
          {
            int sub_size       = cfg_util_sub_size(num_tokens-token_nr, tokens+token_nr, SUB_START_STRING, SUB_END_STRING);
            int sub_size_strip = sub_size - 2;
            int offset         = token_nr + 1;

            if(sub_size_strip < 0)
              sub_size_strip = 0;

            cfg_key_def_type * cfg_key_def_sub = cfg_key_def_alloc_from_tokens(sub_size_strip, tokens+offset, name);
            hash_insert_hash_owned_ref(cfg_type_def->sub_keys, name, cfg_key_def_sub, cfg_key_def_free__);

            token_nr = token_nr + sub_size - 1;
            token = tokens[token_nr];

            scope = get_scope_from_string(token);
            if(scope != SUB_END)
            {
              util_abort("%s: Syntax error, expected \"%s\" got \"%s\".\n", __func__, SUB_END_STRING, token);
            }
          }
          else
          {
            util_abort("%s: Syntax error, expected \"%s\" got \"%s\".\n", __func__, SUB_START_STRING, token);
          }
        }
        else
        {
          util_abort("%s: Syntax error, expected \"%s\" got \"%s\".\n", __func__, SUB_START_STRING, token);
        }
        break;
      }
      case(HELP):
      {
        assert(token_nr + 1 < num_tokens);
        token_nr++;
        token = tokens[token_nr];
        cfg_type_def->help_text = util_alloc_string_copy(token);
        break;
      }
      case(DATA_TYPE):
      {
        util_abort("%s: Syntax error, keyword \"%s\" is only allowed inside \"key\" definitions.\n", __func__, DATA_TYPE_STRING);
        break;
      }
      case(RESTRICTION):
      {
        util_abort("%s: Syntax error, keyword \"%s\" is only allowed inside \"key\" definitions.\n", __func__, RESTRICTION_STRING);
        break;
      }
      default:
        util_abort("%s: Internal error.", __func__);
    }
    token_nr++;
  }
  return cfg_type_def;
}
