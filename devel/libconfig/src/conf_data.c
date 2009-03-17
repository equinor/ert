#include <assert.h>
#include <string.h>
#include <conf_data.h>
#include <util.h>

#define DT_STR_STRING             "string"
#define DT_INT_STRING             "integer"
#define DT_POSINT_STRING          "positive integer"
#define DT_FLOAT_STRING           "floating point number"
#define DT_POSFLOAT_STRING        "positive floating foint number"
#define DT_FILE_STRING            "file"
#define DT_EXEC_STRING            "executable"
#define DT_FOLDER_STRING          "folder"
#define DT_DATE_STRING            "date"



#define RETURN_TYPE_IF_MATCH(STRING,TYPE) if(strcmp(STRING, TYPE ##_STRING) == 0){ return TYPE;}
dt_enum conf_data_get_dt_from_string(
  const char * str)
{
  RETURN_TYPE_IF_MATCH(str, DT_STR);
  RETURN_TYPE_IF_MATCH(str, DT_INT);
  RETURN_TYPE_IF_MATCH(str, DT_POSINT);
  RETURN_TYPE_IF_MATCH(str, DT_FLOAT);
  RETURN_TYPE_IF_MATCH(str, DT_POSFLOAT);
  RETURN_TYPE_IF_MATCH(str, DT_FILE);
  RETURN_TYPE_IF_MATCH(str, DT_EXEC);
  RETURN_TYPE_IF_MATCH(str, DT_FOLDER);
  RETURN_TYPE_IF_MATCH(str, DT_DATE);

  util_abort("%s: Data type \"%s\" is unkown.\n", __func__, str);
  return 0;
}
#undef RETURN_TYPE_IF_MATCH



bool conf_data_string_is_dt(
  const char * str)
{
  if(     !strcmp(str, DT_STR_STRING            )) return true;
  else if(!strcmp(str, DT_INT_STRING            )) return true;
  else if(!strcmp(str, DT_POSINT_STRING         )) return true;
  else if(!strcmp(str, DT_FLOAT_STRING          )) return true;
  else if(!strcmp(str, DT_POSFLOAT_STRING       )) return true;
  else if(!strcmp(str, DT_FILE_STRING           )) return true;
  else if(!strcmp(str, DT_EXEC_STRING           )) return true;
  else if(!strcmp(str, DT_FOLDER_STRING         )) return true;
  else if(!strcmp(str, DT_DATE_STRING           )) return true;
  else                                             return false;
}



const char * conf_data_get_dt_name_ref(
  dt_enum dt)
{
  switch(dt)
  {
    case(DT_STR):
      return DT_STR_STRING;
    case(DT_INT):
      return DT_INT_STRING;
    case(DT_POSINT):
      return DT_POSINT_STRING;
    case(DT_FLOAT):
      return DT_FLOAT_STRING;
    case(DT_POSFLOAT):
      return DT_POSFLOAT_STRING;
    case(DT_FILE):
      return DT_FILE_STRING;
    case(DT_EXEC):
      return DT_EXEC_STRING;
    case(DT_FOLDER):
      return DT_FOLDER_STRING;
    case(DT_DATE):
      return DT_DATE_STRING;
    default:
      util_abort("%s: Internal error.\n", __func__);
      return "";
  }
}



bool conf_data_validate_string_as_dt_value(
  dt_enum      dt,
  const char * str)
{
  if(str == NULL)
    return false;

  switch(dt)
  {
    case(DT_STR):
      return true;
    case(DT_INT):
      return util_sscanf_int(str, NULL);
    case(DT_POSINT):
    {
      int val;
      bool ok = util_sscanf_int(str, &val);
      if(!ok)
        return false;
      else
        return val > 0;
    }
    case(DT_FLOAT):
      return util_sscanf_double(str, NULL);
    case(DT_POSFLOAT):
    {
      double val;
      bool ok = util_sscanf_double(str, &val);
      if(!ok)
        return false;
      else
        return val >= 0.0;
    }
    case(DT_FILE):
    {
      return util_file_exists(str);
    }
    case(DT_EXEC):
    {
      bool ok;
      char * exec = util_alloc_PATH_executable(str);
      ok = exec != NULL;
      free(exec);
      return ok;
    }
    case(DT_FOLDER):
    {
      return util_is_directory(str);
    }
    case(DT_DATE):
    {
      time_t date;
      return util_sscanf_date(str, &date);
    }
    default:
      util_abort("%s: Internal error.\n", __func__);
  }
  return true;
}



bool conf_data_validate_string_as_dt_vector(
  dt_enum      dt,
  const char * str,
  int        * num_elem)
{
  bool ok = true;
  int num_tokens; 
  char ** tokens;

  if(num_elem != NULL)
    *num_elem = 0;

  util_split_string(str, " \t\r\n,", &num_tokens, &tokens);

  for(int i=0; i<num_tokens; i++)
  {
    if(!conf_data_validate_string_as_dt_value(dt, tokens[i]))
      ok = false;
    else if(num_elem != NULL)
      *num_elem = *num_elem + 1;
  }
  util_free_stringlist(tokens, num_tokens);

  return ok;

}



int conf_data_get_int_from_string(
  dt_enum      dt,
  const char * str)
{
  int  value  = 0;
  bool ok     = true;

  switch(dt)
  {
    case(DT_INT):
      ok = util_sscanf_int(str, &value);
      break;
    case(DT_POSINT):
      ok = util_sscanf_int(str, &value);
      break;
    default:
      ok = false;
  }

  if(!ok)
    util_abort("%s: Can not get an int from \"%s\".\n",
               __func__, str);
  
  return value;
}



double conf_data_get_double_from_string(
  dt_enum      dt,
  const char * str)
{
  double value  = 0;
  bool   ok = true;

  switch(dt)
  {
    case(DT_INT):
      ok = util_sscanf_double(str, &value);
      break;
    case(DT_POSINT):
      ok = util_sscanf_double(str, &value);
      break;
    case(DT_FLOAT):
      ok = util_sscanf_double(str, &value);
      break;
    case(DT_POSFLOAT):
      ok = util_sscanf_double(str, &value);
      break;
    default:
      ok = false;
  }

  if(!ok)
    util_abort("%s: Can not get a double from \"%s\".\n",
               __func__, str);
  
  return value;   
}



time_t conf_data_get_time_t_from_string(
  dt_enum      dt,
  const char * str)
{
  time_t value = 0;
  bool   ok    = true;

  switch(dt)
  {
    case(DT_DATE):
      ok = util_sscanf_date(str, &value);
      break;
    default:
      ok = false;
  }

  if(!ok)
    util_abort("%s: Can not get a time_t from \"%s\".\n",
               __func__, str);
  return value;
}



int_vector_type * conf_data_get_int_vector_from_string(
  dt_enum dt,
  const char * str)
{
  int num_tokens; 
  char ** tokens;
  int_vector_type * vec = int_vector_alloc(1,0);

  util_split_string(str, DT_VECTOR_SEP, &num_tokens, &tokens);

  for(int i=0; i<num_tokens; i++)
  {
    int elem = conf_data_get_int_from_string(dt, tokens[i]);
    int_vector_append(vec, elem);
  }
  util_free_stringlist(tokens, num_tokens);

  return vec;
}



double_vector_type * conf_data_get_double_vector_from_string(
  dt_enum dt,
  const char * str)
{
  int num_tokens; 
  char ** tokens;
  double_vector_type * vec = double_vector_alloc(1,0);

  util_split_string(str, DT_VECTOR_SEP, &num_tokens, &tokens);

  for(int i=0; i<num_tokens; i++)
  {
    double elem = conf_data_get_double_from_string(dt, tokens[i]);
    double_vector_append(vec, elem);
  }
  util_free_stringlist(tokens, num_tokens);

  return vec;
}



time_t_vector_type * conf_data_get_time_t_vector_from_string(
  dt_enum dt,
  const char * str)
{
  int num_tokens; 
  char ** tokens;
  time_t_vector_type * vec = time_t_vector_alloc(1,0);

  util_split_string(str, DT_VECTOR_SEP, &num_tokens, &tokens);

  for(int i=0; i<num_tokens; i++)
  {
    time_t elem = conf_data_get_time_t_from_string(dt, tokens[i]);
    time_t_vector_append(vec, elem);
  }
  util_free_stringlist(tokens, num_tokens);

  return vec;
}
