#ifndef ERT_CONFIG_SETTINGS_H
#define ERT_CONFIG_SETTINGS_H

#include <stdbool.h>

#include <ert/util/stringlist.hpp>

#include <ert/config/config_content.hpp>
#include <ert/config/config_parser.hpp>
#include <ert/config/config_schema_item.hpp>

typedef struct config_settings_struct config_settings_type;

config_settings_type *config_settings_alloc(const char *root_key);
void config_settings_free(config_settings_type *settings);
bool config_settings_has_key(const config_settings_type *settings,
                             const char *key);
bool config_settings_set_value(const config_settings_type *config_settings,
                               const char *key, const char *value);
void config_settings_init_parser(const config_settings_type *config_settings,
                                 config_parser_type *config, bool required);
void config_settings_init_parser__(const char *root_key,
                                   config_parser_type *config, bool required);
void config_settings_apply(config_settings_type *config_settings,
                           const config_content_type *config);
stringlist_type *
config_settings_alloc_keys(const config_settings_type *config_settings);

bool config_settings_add_setting(config_settings_type *settings,
                                 const char *key, config_item_types value_type,
                                 const char *initial_value);
void config_settings_add_double_setting(config_settings_type *settings,
                                        const char *key, double initial_value);

double
config_settings_get_double_value(const config_settings_type *config_settings,
                                 const char *key);

bool config_settings_set_value(const config_settings_type *config_settings,
                               const char *key, const char *value);
bool config_settings_set_double_value(
    const config_settings_type *config_settings, const char *key, double value);

#endif
