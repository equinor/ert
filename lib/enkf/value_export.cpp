/*
   Copyright (C) 2017  Equinor ASA, Norway.

   The file 'value_export.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <stdlib.h>
#include <cmath>

#include <map>

#include <ert/util/stringlist.h>
#include <ert/util/double_vector.h>

#include <ert/enkf/value_export.hpp>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <string>
#include <sstream>

#define VALUE_EXPORT_TYPE_ID     5741761


struct value_export_struct {
  UTIL_TYPE_ID_DECLARATION;
  std::string directory;
  std::string base_name;
  std::map<std::string,std::map<std::string, double>> values;

};


static void backup_if_existing(const char * filename) {
    if(not util_file_exists(filename))
        return;
    auto const backup_filename = [filename]() {
        auto const tt = std::chrono::system_clock::to_time_t(
            std::chrono::system_clock::now());
        auto constexpr format = "%Y-%m-%d_%H-%M-%SZ";
        auto fname_stream = std::stringstream();
        fname_stream
            << filename
            << "_backup_"
            << std::put_time(gmtime(&tt), format);
        for(int i = 0;
            util_file_exists(fname_stream.str().c_str()) && i < 100;
            ++i
        ) {
            fname_stream.clear();
            fname_stream
                << filename
                << "_backup_"
                << std::put_time(gmtime(&tt), format)
                << "_"
                << i;
        }
        return fname_stream.str();
    }();
    util_move_file(filename, backup_filename.c_str());
}


value_export_type * value_export_alloc(std::string directory, std::string base_name) {

  value_export_type * value = new value_export_type;
  UTIL_TYPE_ID_INIT( value , VALUE_EXPORT_TYPE_ID );
  value->directory = directory;
  value->base_name = base_name;
  return value;

}


void value_export_free(value_export_type * value) {
  free( value );
}

int value_export_size( const value_export_type * value) {
  int size = 0;
  for(const auto& key_map_pair:value->values)
  {
      size += key_map_pair.second.size();
  }

  return size;
}


void value_export_txt__(const value_export_type * value, const char * filename) {

  if (!value->values.empty()) {
    FILE * stream = util_fopen( filename , "w");
    for (const auto & key_map_pair: value->values) {
      for (const auto &sub_key_value_pair : key_map_pair.second) {
        fprintf(stream, "%s:%s %g\n",  key_map_pair.first.c_str(), sub_key_value_pair.first.c_str(), sub_key_value_pair.second);
      }
    }
    fclose( stream );
  }
}

void value_export_txt(const value_export_type * value) {
  std::string filename = value->directory +"/" + value->base_name + ".txt";
  backup_if_existing(filename.c_str());
  value_export_txt__( value, filename.c_str() );

}

static void generate_hirarchical_keys(const value_export_type * value, FILE * stream)
{
    for (auto iterMaps = value->values.begin(); iterMaps!= value->values.end(); ++iterMaps   ) {
        std::string key = (*iterMaps).first;
        std::map<std::string,double> subMap = (*iterMaps).second;
        fprintf(stream, "\"%s\" : {\n", key.c_str());

        for (auto iterValues = subMap.begin(); iterValues != subMap.end(); ++iterValues) {

            std::string subkey = (*iterValues).first;

            double double_value = (*iterValues).second;
            if (std::isnan(double_value))
                fprintf(stream, "\"%s\" : NaN", subkey.c_str());
            else
                fprintf(stream, "\"%s\" : %g", subkey.c_str(), double_value);


            if (std::next(iterValues) != subMap.end())
                fprintf(stream, ",");
            fprintf(stream, "\n");
        }

        fprintf(stream, "},\n");

    }

}

static void generate_comosite_keys(const value_export_type * value, FILE * stream)
{
    for (auto iterMaps = value->values.begin(); iterMaps!= value->values.end(); ++iterMaps   ) {
        std::string key = (*iterMaps).first;
        std::map<std::string,double> subMap = (*iterMaps).second;

        for (auto iterValues = subMap.begin(); iterValues != subMap.end(); ++iterValues) {

            std::string subkey = (*iterValues).first;

            double double_value = (*iterValues).second;
            if (std::isnan(double_value))
                fprintf(stream, "\"%s\" : NaN", key.c_str());
            else
                fprintf(stream, "\"%s:%s\" : %g", key.c_str(), subkey.c_str(), double_value);

            if (std::next(iterValues) != subMap.end()) {
                fprintf(stream, ",");
                fprintf(stream, "\n");
            }
        }


        if (std::next(iterMaps) != value->values.end())
            fprintf(stream, ",");

        fprintf(stream, "\n");

    }

}

void value_export_json(const value_export_type * value) {
  std::string filename = value->directory +"/" + value->base_name + ".json";
  backup_if_existing(filename.c_str());

  if (!value->values.empty()) {
    FILE * stream = util_fopen( filename.c_str() , "w");
    fprintf(stream, "{\n");
    generate_hirarchical_keys(value, stream);
    generate_comosite_keys(value, stream);
    fprintf(stream, "}\n");
    fclose( stream );
  }

}

void value_export(const value_export_type * value) {
  value_export_txt( value );
  value_export_json( value );
}


void value_export_append(value_export_type * value, const std::string key, const std::string subkey, double double_value){

  if(value->values.find(key) == value->values.end()){
    value->values[key] = std::map<std::string, double>();
  }

  value->values[key][subkey] = double_value;
}

/*****************************************************************/

UTIL_IS_INSTANCE_FUNCTION( value_export , VALUE_EXPORT_TYPE_ID )
