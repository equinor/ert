#include <assert.h>
#include <string.h>
#include <data_type.h>
#include <util.h>

#define DATA_TYPE_STR_STRING      "string"
#define DATA_TYPE_INT_STRING      "int"
#define DATA_TYPE_POSINT_STRING   "posint"
#define DATA_TYPE_FLOAT_STRING    "float"
#define DATA_TYPE_POSFLOAT_STRING "posfloat"
#define DATA_TYPE_FILE_STRING     "file"
#define DATA_TYPE_DATE_STRING     "date"

#define RETURN_TYPE_IF_MATCH(STRING,TYPE) if(strcmp(STRING, TYPE ##_STRING) == 0){ return TYPE;}
data_type_enum get_data_type_from_string(const char * str)
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



bool is_data_type(const char * str)
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



const char * get_data_type_str_ref(data_type_enum data_type)
{
  switch(data_type)
  {
    case(DATA_TYPE_STR):
      return DATA_TYPE_STR_STRING;
    case(DATA_TYPE_INT):
      return DATA_TYPE_INT_STRING;
    case(DATA_TYPE_POSINT):
      return DATA_TYPE_POSINT_STRING;
    case(DATA_TYPE_FLOAT):
      return DATA_TYPE_FLOAT_STRING;
    case(DATA_TYPE_POSFLOAT):
      return DATA_TYPE_POSFLOAT_STRING;
    case(DATA_TYPE_FILE):
      return DATA_TYPE_FILE_STRING;
    case(DATA_TYPE_DATE):
      return DATA_TYPE_DATE_STRING;
    default:
      util_abort("%s: Internal error.\n", __func__);
      return "";
  }
}



bool validate_str_as_data_type( data_type_enum data_type, const char * str)
{
  assert(str != NULL);
  switch(data_type)
  {
    case(DATA_TYPE_STR):
      return true;
      break;
    case(DATA_TYPE_INT):
      //TODO Implement me.
      break;
    case(DATA_TYPE_POSINT):
      //TODO Implement me.
      break;
    case(DATA_TYPE_FLOAT):
      //TODO Implement me.
      break;
    case(DATA_TYPE_POSFLOAT):
      //TODO Implement me.
      break;
    case(DATA_TYPE_FILE):
      //TODO Implement me.
      break;
    case(DATA_TYPE_DATE):
      //TODO Implement me.
      break;
    default:
      util_abort("%s: Internal error.\n", __func__);
  }
  printf("%s: WARNING! Validation of data types is not fully implemented!.\n", __func__);
  return true;
}
