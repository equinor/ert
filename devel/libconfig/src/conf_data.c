#include <assert.h>
#include <string.h>
#include <conf_data.h>
#include <util.h>

#define DT_STR_STRING             "string"
#define DT_INT_STRING             "integer"
#define DT_POSINT_STRING          "positive integer"
#define DT_FLOAT_STRING           "floating point number"
#define DT_POSFLOAT_STRING        "positive floating foint number"
#define DT_INT_VECTOR_STRING      "integer vector"
#define DT_POSINT_VECTOR_STRING   "positive integer vector"
#define DT_FLOAT_VECTOR_STRING    "float vector"
#define DT_POSFLOAT_VECTOR_STRING "positive float vector"
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
  RETURN_TYPE_IF_MATCH(str, DT_INT_VECTOR);
  RETURN_TYPE_IF_MATCH(str, DT_POSINT_VECTOR);
  RETURN_TYPE_IF_MATCH(str, DT_FLOAT_VECTOR);
  RETURN_TYPE_IF_MATCH(str, DT_POSFLOAT_VECTOR);
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
  else if(!strcmp(str, DT_INT_VECTOR_STRING     )) return true;
  else if(!strcmp(str, DT_POSINT_VECTOR_STRING  )) return true;
  else if(!strcmp(str, DT_FLOAT_VECTOR_STRING   )) return true;
  else if(!strcmp(str, DT_POSFLOAT_VECTOR_STRING)) return true;
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
    case(DT_INT_VECTOR):
      return DT_INT_VECTOR_STRING;
    case(DT_POSINT_VECTOR):
      return DT_POSINT_VECTOR_STRING;
    case(DT_FLOAT_VECTOR):
      return DT_FLOAT_VECTOR_STRING;
    case(DT_POSFLOAT_VECTOR):
      return DT_POSFLOAT_VECTOR_STRING;
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


static
bool validate_as_integer_vector(
  const char * str,
  bool         positive,
  int_vector_type * vector)
{
  bool ok = true;
  int  value;
  int num_tokens; 
  char ** tokens;

  util_split_string(str, " \t\r\n,", &num_tokens, &tokens);

  for(int i=0; i<num_tokens; i++)
  {
    if(!util_sscanf_int(tokens[i], &value))
      ok = false;
    else if(positive)
    {
      if(value < 0)
        ok = false;
    }
    if(vector !=NULL && ok)
      int_vector_append(vector, value);
  }
  util_free_stringlist(tokens, num_tokens);

  return ok;
}



static
bool validate_as_double_vector(
  const char * str,
  bool         positive,
  double_vector_type * vector)
{
  bool ok = true;
  double  value;
  int num_tokens; 
  char ** tokens;

  util_split_string(str, " \t\r\n,", &num_tokens, &tokens);

  for(int i=0; i<num_tokens; i++)
  {
    if(!util_sscanf_double(tokens[i], &value))
      ok = false;
    else if(positive)
    {
      if(value < 0.0)
        ok = false;
    }
    if(vector != NULL && ok)
      double_vector_append(vector, value);
  }
  util_free_stringlist(tokens, num_tokens);

  return ok;
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
    case(DT_INT_VECTOR):
    {
      return validate_as_integer_vector(str, false, NULL);
    }
    case(DT_POSINT_VECTOR):
    {
      return validate_as_integer_vector(str, true, NULL);
    }
    case(DT_FLOAT_VECTOR):
    {
      return validate_as_double_vector(str, false, NULL);
    }
    case(DT_POSFLOAT_VECTOR):
    {
      return validate_as_double_vector(str, true, NULL);
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
  dt_enum      dt,
  const char * str)
{
  bool ok = true;
  int_vector_type * vector = int_vector_alloc(1, 0);
  switch(dt)
  {
    case(DT_INT_VECTOR):
      ok = validate_as_integer_vector(str, false, vector);
      break;
    case(DT_POSINT_VECTOR):
      ok = validate_as_integer_vector(str, true, vector);
      break;
    default:
      ok = false;
  }
  if(!ok)
    util_abort("%s: Can not get a int vector from \"%s\".\n",
               __func__, str);
  return vector;
}



double_vector_type * conf_data_get_dobule_vector_from_string(
  dt_enum      dt,
  const char * str)
{
  bool ok = true ;
  double_vector_type * vector = double_vector_alloc(1, 0.0);
  switch(dt)
  {
    case(DT_FLOAT_VECTOR):
      ok = validate_as_double_vector(str, false, vector);
      break;
    case(DT_POSFLOAT_VECTOR):
      ok = validate_as_double_vector(str, true, vector);
      break;
    default:
      ok = false;
  }
  if(!ok)
    util_abort("%s: Can not get a double vector from \"%s\".\n",
               __func__, str);
  return vector;
}
