/*
   Copyright (C) 2012  Statoil ASA, Norway. 
    
   The file 'config_get.c' is part of ERT - Ensemble based Reservoir Tool. 
    
   ERT is free software: you can redistribute it and/or modify 
   it under the terms of the GNU General Public License as published by 
   the Free Software Foundation, either version 3 of the License, or 
   (at your option) any later version. 
    
   ERT is distributed in the hope that it will be useful, but WITHOUT ANY 
   WARRANTY; without even the implied warranty of MERCHANTABILITY or 
   FITNESS FOR A PARTICULAR PURPOSE.   
    
   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html> 
   for more details. 
*/


/*****************************************************************/
/* All the functions in this block will operate on the last item 
   which has been set with a particular key value. So assuming the 
   config file looks like:

   KEY   VALUE1
   KEY   VALUE2  OPTIONAL 
   KEY   100     VALUE3   OPTIONAL  ERROR

   these functions will all operate on the last line in the config file:

             KEY 100 VALUE3 OPTIONAL ERROR
*/


static void assert_key_value(const config_schema_item_type * item) {
  int argc_min , argc_max;
  config_schema_item_get_argc( item , &argc_min , &argc_max);

  if (!((argc_min == 1) && (argc_min == 1)))
    util_abort("%s: item:%s before calling config_get_value() functions *without* index you must set argc_min == argc_max = 1 \n",__func__ , config_schema_item_get_kw( item ));
}



bool config_get_value_as_bool(const config_type * config , const char * kw) {
  config_content_item_type * item = config_get_content_item(config , kw);
  config_content_node_type * node = config_content_item_get_last_node( item );
  assert_key_value( item->schema );
  return config_content_node_iget_as_bool(node , 0);
}

int config_get_value_as_int(const config_type * config , const char * kw) {
  config_content_item_type * item = config_get_content_item(config , kw);
  config_content_node_type * node = config_content_item_get_last_node( item );
  assert_key_value( item->schema );
  return config_content_node_iget_as_int(node , 0);
}

double config_get_value_as_double(const config_type * config , const char * kw) {
  config_content_item_type * item = config_get_content_item(config , kw);
  config_content_node_type * node = config_content_item_get_last_node( item );
  assert_key_value( item->schema );
  return config_content_node_iget_as_double(node , 0);
}

const char * config_get_value(const config_type * config , const char * kw) {
  config_content_item_type * item = config_get_content_item(config , kw);
  config_content_node_type * node = config_content_item_get_last_node( item );
  assert_key_value( item->schema );
  return config_content_node_iget(node , 0);
}

