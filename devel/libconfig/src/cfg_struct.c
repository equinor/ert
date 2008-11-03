#include <assert.h>
#include <string.h>
#include <stdbool.h>
#include <hash.h>
#include <util.h>
#include <cfg_struct.h>
#include <cfg_util.h>
#include <data_type.h>


/**
  The purpose of this library is to provide lightweight configuration framework that:

    1. Provides a flexible and unified syntax for all configuration files.
    2. Checks input data, both wrt. to data type and required input.
    3. Can use default values, thus requiring minimal user input.
    4. Can provide extensive help to the user on missing items, invalid values etc.


  The library is used in the following way:

    1. The developer writes a "configuration specification", which specifies requirements
       to the user provided configuration file. This can be done in the following ways:

       I)  Writing a configuration specification file using a minimal, built-in language.
           This file is then loaded run-time, before the user-provided configuration file.
       II) Hard-coding the configuration specification in the code before the user provided
           configuration file is loaded.
       
       Using the configuration specification, the developer can create two different types.

       structs: A struct can hold other structs and items. In other words, it can be
                both a tree node and a leaf. However, it cannot hold any data other
                than its name. In the user provided configuration file, multiple structs
                of the same type can exist in the same scope, provided they have different
                names. Using the keyword "required" before a struct definition, will check
                that a user provided configuration file have declared at least one 
                struct of the desired type.

        items:  An item cannot hold structs or other items, it is a leaf node. It is
                not allowed to have a struct definition and an item definition with
                the same name in the same scope. When defining the item, is is possible
                to require the user provided value of the item to be parseable to 
                many different types. For example, one may require that it is parseable
                to a positive integer, or to the name of an existing file. Using the
                keyword "required" before the definition of the item, will check
                that the user has provided a value for the item, or that a default
                value exists.


    2. Using the configuration specification and code from the library, the developer
       can internalize a user provided configuration file. This is done as follows:

       Code example:
       /-------------------
       | cfg_struct_def_type * cfg_spec = .... ;
       | cfg_struct_type     * user_cfg = cfg_struct_alloc_from_file("config.txt", cfg_spec);
                
       If cfg_struct_alloc_from_file fails to parse a configuration according to the
       given specification, it will print an abort message to the user and quit. Thus,
       on success, the developer can be certain that all required items and structs are set
       and that values are parseable according to their specification. Thus, there is little
       need for checking the values in "user_cfg".
*/


struct cfg_struct_struct
{
  char * struct_type_name;
  char * name;

  hash_type * sub_items;        /* Hash indexed with items. All values are strings. */
  hash_type * sub_structs;      /* Hash indexed with names of struct instances. All values are cfg_struct_types. */
};



/*
  These static functions are called before their implementation. Thus, header info is needed.
*/
static cfg_struct_type * cfg_struct_alloc_from_buffer(const cfg_struct_def_type *, char **, char **, const char *, const char *, set_type *);





/*
  NOTE: This is *not* the same prim_enum as in cfg_struct_def. The primitives are simply different
        when parsing a configuration spec and a given configuration.
*/

#define STR_CFG_ASSIGN      "="
#define STR_CFG_SCOPE_START "{"
#define STR_CFG_SCOPE_STOP  "}"
#define STR_CFG_END         ";"
#define STR_CFG_INCLUDE     "include"
#define STR_CFG_RESET       "reset"
#define STR_CFG_PAR_START   "("
#define STR_CFG_PAR_END     ")"



typedef enum {CFG_ASSIGN, CFG_SCOPE_START, CFG_SCOPE_STOP, CFG_END, CFG_INCLUDE, CFG_RESET, CFG_PAR_START, CFG_PAR_END, CFG_STRUCT, CFG_ITEM} prim_enum;



#define RETURN_PRIM_IF_MATCH_MACRO(PRIM, STRING) if(!strcmp(STRING, STR_## PRIM)){return PRIM;}
static prim_enum get_prim_from_str(const cfg_struct_def_type * cfg_struct_def, const char * str)
{
  if(str == NULL)
    util_abort("%s: Trying to valdiate NULL as a token.\n");

  RETURN_PRIM_IF_MATCH_MACRO(CFG_ASSIGN, str);
  RETURN_PRIM_IF_MATCH_MACRO(CFG_SCOPE_START, str);
  RETURN_PRIM_IF_MATCH_MACRO(CFG_SCOPE_STOP, str);
  RETURN_PRIM_IF_MATCH_MACRO(CFG_END, str);
  RETURN_PRIM_IF_MATCH_MACRO(CFG_INCLUDE, str);
  RETURN_PRIM_IF_MATCH_MACRO(CFG_RESET, str);
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
static bool str_is_prim(const cfg_struct_def_type * cfg_struct_def, const char * str)
{
  if(str == NULL)
    util_abort("%s: Trying to valdiate NULL as a token.\n");

  RETURN_TRUE_IF_MATCH_MACRO(CFG_ASSIGN, str);
  RETURN_TRUE_IF_MATCH_MACRO(CFG_SCOPE_START, str);
  RETURN_TRUE_IF_MATCH_MACRO(CFG_SCOPE_STOP, str);
  RETURN_TRUE_IF_MATCH_MACRO(CFG_END, str);
  RETURN_TRUE_IF_MATCH_MACRO(CFG_INCLUDE, str);
  RETURN_TRUE_IF_MATCH_MACRO(CFG_RESET, str);
  RETURN_TRUE_IF_MATCH_MACRO(CFG_PAR_START, str);
  RETURN_TRUE_IF_MATCH_MACRO(CFG_PAR_END, str);

  if(hash_has_key(cfg_struct_def->sub_items, str))
    return true;
  if(hash_has_key(cfg_struct_def->sub_structs, str))
    return true;

  return false;
}
#undef RETURN_TRUE_IF_MATCH_MACRO



/*
  Returns true if token is the string corresponding to the prim_enum prim (i.e. "include" -> CFG_INCLUDE).
*/
static bool validate_token(const cfg_struct_def_type * cfg_struct_def, prim_enum prim, const char * token)
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



/*
  This function checks that:
  1. All required items are set.
  2. All required sub structs are set.
  3. That all items obey their data type.
  4. That all items obey their restriction set.
  5. Repeats 1-4 for all set sub structs.

  Thus, it is suffcient to call this on the root to verify the entire config tree.
*/
static bool cfg_struct_validate(const cfg_struct_def_type * cfg_struct_def, const cfg_struct_type * cfg_struct)
{
  bool ok = true;
  /* Check that required items are set. */
  int     num_req_items = set_get_size(cfg_struct_def->required_sub_items);
  char ** req_items     = set_alloc_keylist(cfg_struct_def->required_sub_items);

  for(int req_item_nr = 0; req_item_nr < num_req_items; req_item_nr++)
  {
    if(!hash_has_key(cfg_struct->sub_items, req_items[req_item_nr]))
    {
      printf("\nERROR: Required item \"%s\" has not been set in struct \"%s\" of type \"%s\".\n", req_items[req_item_nr], cfg_struct->name, cfg_struct->struct_type_name);
      cfg_item_def_type * cfg_item_def = hash_get(cfg_struct_def->sub_items, req_items[req_item_nr]);
      cfg_item_def_printf_help(cfg_item_def);
      ok = false;
    }  
  }

  util_free_stringlist(req_items, num_req_items);


  /* Check that all items obey their restriction set and is parseable to their data_type. */
  int num_items = hash_get_size(cfg_struct->sub_items);
  char ** items = hash_alloc_keylist(cfg_struct->sub_items);

  for(int item_nr = 0; item_nr < num_items; item_nr++)
  {
    const char * key     = items[item_nr];
    const char * value   = hash_get_string(cfg_struct->sub_items, items[item_nr]);
    cfg_item_def_type * cfg_item_def = hash_get(cfg_struct_def->sub_items, key);

    if(!validate_str_as_data_type(cfg_item_def->data_type, value))
    {
      printf("\nERROR: Failed to get data or file/folder of type \"%s\" for item \"%s\" from \"%s\".\n", get_data_type_str_ref(cfg_item_def->data_type), key, value);
      cfg_item_def_printf_help(cfg_item_def);
      ok = false;
    }

    int num_restrictions = set_get_size(cfg_item_def->restriction);
    if( num_restrictions > 0)
    {
      if(!set_has_key(cfg_item_def->restriction, value))
      {
        printf("\nERROR: \"%s\" is not a valid value for \"%s\".\n", value, key);
        cfg_item_def_printf_help(cfg_item_def);
      }
    }

  }


  util_free_stringlist(items, num_items);

  /* Check that required structs are set. */
  set_type * sub_struct_types = cfg_struct_alloc_sub_structs_type_set(cfg_struct);
  int     num_req_structs = set_get_size(cfg_struct_def->required_sub_structs);
  char ** req_structs     = set_alloc_keylist(cfg_struct_def->required_sub_structs);

  for(int req_struct_nr = 0; req_struct_nr < num_req_structs; req_struct_nr++)
  {
    if(!set_has_key(sub_struct_types, req_structs[req_struct_nr]))
    {
      printf("\nERROR: Required struct \"%s\" has not been set in struct \"%s\" of type \"%s\".\n", req_structs[req_struct_nr], cfg_struct->name, cfg_struct->struct_type_name);
      cfg_struct_def_type * cfg_struct_def_sub = hash_get(cfg_struct_def->sub_structs, req_structs[req_struct_nr]);
      if(cfg_struct_def_sub->help != NULL)
        printf("\n       Help on struct \"%s\":\n       %s\n\n", req_structs[req_struct_nr], cfg_struct_def_sub->help);
      ok = false;
    }
  }

  util_free_stringlist(req_structs, num_req_structs);
  set_free(sub_struct_types);

  /* Validate all sub structs. */
  int     num_sub_structs = hash_get_size(cfg_struct->sub_structs);
  char ** sub_structs     = hash_alloc_keylist(cfg_struct->sub_structs);

  for(int sub_struct_nr = 0; sub_struct_nr < num_sub_structs; sub_struct_nr++)
  {
    cfg_struct_type     * cfg_struct_sub     = hash_get(cfg_struct->sub_structs, sub_structs[sub_struct_nr]);
    cfg_struct_def_type * cfg_struct_def_sub = hash_get(cfg_struct_def->sub_structs, cfg_struct_sub->struct_type_name);
    if(!cfg_struct_validate(cfg_struct_def_sub, cfg_struct_sub))
      ok = false;
  }

  util_free_stringlist(sub_structs, num_sub_structs);

  return ok;
}



/**************************************************************************************************************************************/



static void cfg_struct_set_item
(
          const cfg_struct_def_type * cfg_struct_def,
          const char * item,
          char ** __buffer_pos,
          cfg_struct_type * cfg_struct
)
{
  char * token = cfg_util_alloc_next_token(__buffer_pos);
  if(!validate_token(cfg_struct_def, CFG_ASSIGN, token))
    util_abort("%s: Syntax error. Expected \"%s\", got \"%s\".\n", __func__, STR_CFG_ASSIGN, token);

  /*
    Just add the token now. We will check that it obeys the data_type later.
  */
  free(token);
  token = cfg_util_alloc_next_token(__buffer_pos);
  if(str_is_prim(cfg_struct_def, token))
    util_abort("%s: Syntax error. Expected a data value. Got primitive \"%s\".\n", __func__, token);
  hash_insert_string(cfg_struct->sub_items, item, token);

  free(token);
  token = cfg_util_alloc_next_token(__buffer_pos);
  if(!validate_token(cfg_struct_def, CFG_END, token))
    util_abort("%s: Syntax error. Expected \"%s\", got \"%s\".\n", __func__, STR_CFG_END, token);
  free(token);
}



static void cfg_struct_alloc_sub_struct
(
            const cfg_struct_def_type * cfg_struct_def,
            char ** __buffer,
            char ** __buffer_pos,
            const char * struct_type_name,
            set_type * src_files,
            cfg_struct_type * cfg_struct
)
{
  char * name = cfg_util_alloc_next_token(__buffer_pos);
  if(str_is_prim(cfg_struct_def, name))
    util_abort("%s: Syntax error. Expected a string. Got primitive \"%s\".\n", __func__, name);

  if(hash_has_key(cfg_struct->sub_structs, name))
    util_abort("%s: Error, identifier \"%s\" has already been used.\n", __func__, name);

  cfg_struct_def_type * cfg_struct_def_sub = hash_get(cfg_struct_def->sub_structs, struct_type_name);
  cfg_struct_type * cfg_struct_sub = cfg_struct_alloc_from_buffer(cfg_struct_def_sub, __buffer, __buffer_pos, struct_type_name, name, src_files);

  hash_insert_hash_owned_ref(cfg_struct->sub_structs, name, cfg_struct_sub, cfg_struct_free__);
  free(name);
    
}



/*
  This function will add items which have a default value set in the definition.
*/
static void cfg_struct_add_defaults(const cfg_struct_def_type * cfg_struct_def, cfg_struct_type * cfg_struct)
{
  int     num_items = hash_get_size(cfg_struct_def->sub_items);
  char ** items = hash_alloc_keylist(cfg_struct_def->sub_items);

  for(int item_nr = 0; item_nr < num_items; item_nr++)
  { 
    cfg_item_def_type * cfg_item_def = hash_get(cfg_struct_def->sub_items, items[item_nr]);
    if(cfg_item_def->default_value != NULL)
    {
      hash_insert_string(cfg_struct->sub_items, items[item_nr], cfg_item_def->default_value);
    }
  }  
  util_free_stringlist(items, num_items);

}




static cfg_struct_type * cfg_struct_alloc_from_buffer
(
                        const cfg_struct_def_type * cfg_struct_def,
                        char ** __buffer,
                        char ** __buffer_pos,
                        const char * struct_type_name,
                        const char * name,
                        set_type * src_files
)
{
  assert(name != NULL);

  cfg_struct_type * cfg_struct = util_malloc(sizeof * cfg_struct, __func__);
  
  cfg_struct->struct_type_name = util_alloc_string_copy(struct_type_name);
  cfg_struct->name             = util_alloc_string_copy(name);
  cfg_struct->sub_items        = hash_alloc();
  cfg_struct->sub_structs      = hash_alloc();

  /*
    Add defaults.
  */
  cfg_struct_add_defaults(cfg_struct_def, cfg_struct);



  bool scope_start_set = false;
  bool scope_end_set   = false;
  bool struct_finished = false;

  for(;;)
  {
    char * token = cfg_util_alloc_next_token(__buffer_pos);

    /*
      First, check if something has gone haywire or if we are at end of buffer.
    */
    if(token == NULL && scope_start_set != scope_end_set)
      util_abort("%s: Syntax error in struct \"%s\". Unexpected end of file.\n", __func__, name);
    else if(token == NULL)
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
        if(!scope_start_set)
          struct_finished = true; 
        else
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
    util_safe_free(token);


    if(struct_finished && scope_end_set)
    {
      /*
        We've seen the closing "}" of the struct (or EOF).
        Check if it's followed by a ";", which is required.
      */
      char * buffer_pos = *__buffer_pos;
      token = cfg_util_alloc_next_token(&buffer_pos);
      if(token == NULL)
      {
        util_abort("%s: Syntax error in struct \"%s\". Expected \"%s\", got EOF (end of file).\n", __func__, name, STR_CFG_END);
      }
      else if(validate_token(cfg_struct_def, CFG_END, token))
      {
        *__buffer_pos = buffer_pos;
      }
      else
      {
        util_abort("%s: Syntax error in struct \"%s\". Expected \"%s\", got \"%s\".\n", __func__, name, STR_CFG_END, token);
      }
      free(token);
      break;
    }
    else if(struct_finished && !scope_end_set)
    {
      break;
    }
  }
  if(scope_start_set != scope_end_set)
    util_abort("%s: Syntax error in struct \"%s\". Could not match delimiters.\n", __func__, name); 

  return cfg_struct;
}



/**************************************************************************************************************************************/



cfg_struct_type * cfg_struct_alloc_from_file(const char * filename, const cfg_struct_def_type * cfg_struct_def)
{
  char * pad_keys[] = {"{","}","=",";","(",")"};
  char * buffer = cfg_util_fscanf_alloc_token_buffer(filename, "--", 6, (const char **) pad_keys);
  char * buffer_pos = buffer;

  // This is used to avoid infinite inclusion.
  set_type * src_files = set_alloc_empty();
  // TODO Should add absolute path.
  set_add_key(src_files, filename);

  cfg_struct_type * cfg_struct  = cfg_struct_alloc_from_buffer(cfg_struct_def, &buffer, &buffer_pos, "root", "root", src_files);
  free(buffer);

  /* Validate the struct. */
  if(!cfg_struct_validate(cfg_struct_def, cfg_struct))
    util_abort("%s: Failed to validate configuration.\n", __func__);

  /* Free the src_files set. */
  set_free(src_files);

  return cfg_struct;
}



const char * cfg_struct_get_name(const cfg_struct_type * cfg_struct)
{
  return cfg_struct->name;
}



const char * cfg_struct_get_struct_type_name(const cfg_struct_type * cfg_struct)
{
  return cfg_struct->struct_type_name;
}



void cfg_struct_free(cfg_struct_type * cfg_struct)
{
  free(cfg_struct->struct_type_name);
  free(cfg_struct->name);
  hash_free(cfg_struct->sub_items);
  hash_free(cfg_struct->sub_structs);
  free(cfg_struct);
}



void cfg_struct_free__(void * cfg_struct)
{
  cfg_struct_free( (cfg_struct_type *) cfg_struct);
}



bool cfg_struct_has_item(const cfg_struct_type * cfg_struct, const char * item_name)
{
  return hash_has_key(cfg_struct->sub_items, item_name);
}



const char * cfg_struct_get_item(const cfg_struct_type * cfg_struct, const char * item_name)
{
  if(!cfg_struct_has_item(cfg_struct, item_name))
    util_abort("%s: Struct \"%s\" of type \"%s\" has no item with name \"%s\".\n", __func__, cfg_struct->name, cfg_struct->struct_type_name, item_name);
  return hash_get(cfg_struct->sub_items, item_name);
}



bool cfg_struct_has_sub_struct(const cfg_struct_type * cfg_struct, const char * struct_name)
{
  return hash_has_key(cfg_struct->sub_structs, struct_name);
}



const char * cfg_struct_get_sub_struct_type(const cfg_struct_type * cfg_struct, const char * struct_name)
{
  if(!cfg_struct_has_sub_struct(cfg_struct, struct_name))
    util_abort("The struct \"%s\" of type \"%s\" has no sub struct with name \"%s\".\n", cfg_struct->name, cfg_struct->struct_type_name, struct_name);

  cfg_struct_type * cfg_struct_sub = hash_get(cfg_struct->sub_structs, struct_name);
  return cfg_struct_sub->struct_type_name;
}



int cfg_struct_num_instances_of_sub_struct_type(const cfg_struct_type * cfg_struct, const char * sub_struct_type_name)
{
  int     num_occurences  = 0;
  int     num_sub_structs = hash_get_size(cfg_struct->sub_structs);
  char ** sub_structs     = hash_alloc_keylist(cfg_struct->sub_structs);

  for(int sub_struct_nr = 0; sub_struct_nr < num_sub_structs; sub_struct_nr++)
  {
    cfg_struct_type * cfg_struct_sub = hash_get(cfg_struct->sub_structs, sub_structs[sub_struct_nr]);
    if(strcmp(cfg_struct_sub->struct_type_name, sub_struct_type_name) == 0) 
      num_occurences++;
  }
  util_free_stringlist(sub_structs, num_sub_structs);
  return num_occurences;
}



set_type * cfg_struct_alloc_sub_structs_type_set(const cfg_struct_type * cfg_struct)
{
  set_type * sub_structs_type_set = set_alloc_empty();
  int     num_sub_structs = hash_get_size(cfg_struct->sub_structs);
  char ** sub_structs     = hash_alloc_keylist(cfg_struct->sub_structs);

  for(int sub_struct_nr = 0; sub_struct_nr < num_sub_structs; sub_struct_nr++)
  {
    cfg_struct_type * cfg_struct_sub = hash_get(cfg_struct->sub_structs, sub_structs[sub_struct_nr]);
    if(!set_has_key(sub_structs_type_set, cfg_struct_sub->struct_type_name))
      set_add_key(sub_structs_type_set, cfg_struct_sub->struct_type_name);
  }
  util_free_stringlist(sub_structs, num_sub_structs);
  return sub_structs_type_set;
}



stringlist_type * cfg_struct_alloc_instances_of_sub_struct_type(const cfg_struct_type * cfg_struct, const char * sub_struct_type_name)
{
  stringlist_type * stringlist = stringlist_alloc_new();

  int     num_sub_structs = hash_get_size(cfg_struct->sub_structs);
  char ** sub_structs     = hash_alloc_keylist(cfg_struct->sub_structs);
  for(int sub_struct_nr = 0; sub_struct_nr < num_sub_structs; sub_struct_nr++)
  {
    cfg_struct_type * cfg_struct_sub = hash_get(cfg_struct->sub_structs, sub_structs[sub_struct_nr]);
    if(strcmp(cfg_struct_sub->struct_type_name, sub_struct_type_name) == 0) 
      stringlist_append_copy(stringlist, sub_structs[sub_struct_nr]);
  }
  util_free_stringlist(sub_structs, num_sub_structs);

  return stringlist;
}



cfg_struct_type * cfg_struct_get_sub_struct(const cfg_struct_type * cfg_struct, const char * struct_name)
{
  if(!cfg_struct_has_sub_struct(cfg_struct, struct_name))
    util_abort("The struct \"%s\" of type \"%s\" has no sub struct with name \"%s\".\n", cfg_struct->name, cfg_struct->struct_type_name, struct_name);

  cfg_struct_type * cfg_struct_sub = hash_get(cfg_struct->sub_structs, struct_name);
  return cfg_struct_sub;
}



