/*
   Copyright (C) 2012  Statoil ASA, Norway. 
    
   The file 'ext_script.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#include <stdbool.h>
#include <stdlib.h>
#include <dlfcn.h>

#include <int_vector.h>
#include <util.h>
#include <type_macros.h>
#include <stringlist.h>
#include <arg_pack.h>
#include <vector.h>

#include <config.h>

#include <ext_cmd.h>
#include <ext_script.h>



struct cmd_struct {
  const ext_cmd_type * ext_cmd;
  stringlist_type    * arglist;
};



struct ext_script_struct {
  config_type * config_compiler;
  vector_type * cmd_list;
};
