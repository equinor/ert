#ifndef VALUE_EXPORT_H
#define VALUE_EXPORT_H

#include <string>

typedef struct value_export_struct value_export_type;

void value_export_free(value_export_type *value);
value_export_type *value_export_alloc(std::string directory,
                                      std::string base_name);
int value_export_size(const value_export_type *value);
void value_export_json(const value_export_type *value);
void value_export_txt(const value_export_type *value);
void value_export_txt__(const value_export_type *value, const char *filename);
void value_export(const value_export_type *value);
void value_export_append(value_export_type *value, const std::string key,
                         const std::string subkey, double double_value);

#endif
