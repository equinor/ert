/*
   Copyright (C) 2017  Statoil ASA, Norway.

   The file 'value_export.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef VALUE_EXPORT_H
#define VALUE_EXPORT_H
#ifdef __cplusplus
extern "C" {
#endif

#include <ert/util/type_macros.h>

  typedef struct value_export_struct value_export_type;

  void                 value_export_free(value_export_type * export);
  value_export_type * value_export_alloc(const char * directory, const char * base_name);
  int                  value_export_size(const value_export_type * export);
  void                 value_export_json(const value_export_type * export);
  void                 value_export_txt(const value_export_type * export);
  void                 value_export_txt__(const value_export_type * export, const char * filename);
  void                 value_export(const value_export_type * export);
  void                 value_export_append( value_export_type * export, const char * key , double value);

UTIL_IS_INSTANCE_HEADER(value_export);

#ifdef __cplusplus
}
#endif
#endif
