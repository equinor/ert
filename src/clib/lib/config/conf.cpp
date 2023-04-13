#include <filesystem>
#include <iostream>
#include <vector>

#include <assert.h>
#include <string.h>

#include <ert/logging.hpp>
#include <ert/util/path_stack.hpp>

#include <ert/config/conf.hpp>
#include <ert/config/conf_util.hpp>

namespace fs = std::filesystem;

static auto logger = ert::get_logger("config");

conf_class_type *conf_class_alloc_empty(const char *class_name,
                                        bool require_instance, bool singleton,
                                        const char *help) {
    assert(class_name != NULL);

    conf_class_type *conf_class =
        (conf_class_type *)util_malloc(sizeof *conf_class);

    conf_class->super_class = NULL;
    conf_class->class_name = util_alloc_string_copy(class_name);
    conf_class->help = NULL;
    conf_class->require_instance = require_instance;
    conf_class->singleton = singleton;
    conf_class->sub_classes = hash_alloc();
    conf_class->item_specs = hash_alloc();
    conf_class->item_mutexes = vector_alloc_new();

    conf_class_set_help(conf_class, help);
    return conf_class;
}

void conf_class_free(conf_class_type *conf_class) {
    free(conf_class->class_name);
    free(conf_class->help);
    hash_free(conf_class->sub_classes);
    hash_free(conf_class->item_specs);
    vector_free(conf_class->item_mutexes);
    free(conf_class);
}

void conf_class_free__(void *conf_class) {
    conf_class_free((conf_class_type *)conf_class);
}

static conf_instance_type *
conf_instance_alloc_default(const conf_class_type *conf_class,
                            const char *name) {
    assert(conf_class != NULL);
    assert(name != NULL);

    conf_instance_type *conf_instance =
        (conf_instance_type *)util_malloc(sizeof *conf_instance);

    conf_instance->conf_class = conf_class;
    conf_instance->name = util_alloc_string_copy(name);
    conf_instance->sub_instances = hash_alloc();
    conf_instance->items = hash_alloc();
    {
        /* Insert items that have a default value in their specs. */
        int num_item_specs = hash_get_size(conf_class->item_specs);
        char **item_spec_keys = hash_alloc_keylist(conf_class->item_specs);

        for (int item_spec_nr = 0; item_spec_nr < num_item_specs;
             item_spec_nr++) {
            const char *item_spec_name = item_spec_keys[item_spec_nr];
            const conf_item_spec_type *conf_item_spec =
                (const conf_item_spec_type *)hash_get(conf_class->item_specs,
                                                      item_spec_name);
            if (conf_item_spec->default_value != NULL) {
                conf_item_type *conf_item = conf_item_alloc(
                    conf_item_spec, conf_item_spec->default_value);
                conf_instance_insert_owned_item(conf_instance, conf_item);
            }
        }

        util_free_stringlist(item_spec_keys, num_item_specs);
    }

    return conf_instance;
}

void conf_instance_free(conf_instance_type *conf_instance) {
    free(conf_instance->name);
    hash_free(conf_instance->sub_instances);
    hash_free(conf_instance->items);
    free(conf_instance);
}

void conf_instance_free__(void *conf_instance) {
    conf_instance_free((conf_instance_type *)conf_instance);
}

conf_item_spec_type *conf_item_spec_alloc(const char *name, bool required_set,
                                          dt_enum dt, const char *help) {
    assert(name != NULL);

    conf_item_spec_type *conf_item_spec =
        (conf_item_spec_type *)util_malloc(sizeof *conf_item_spec);

    conf_item_spec->super_class = NULL;
    conf_item_spec->name = util_alloc_string_copy(name);
    conf_item_spec->required_set = required_set;
    conf_item_spec->dt = dt;
    conf_item_spec->default_value = NULL;
    conf_item_spec->help = NULL;
    conf_item_spec->help = util_realloc_string_copy(conf_item_spec->help, help);

    conf_item_spec->restriction = new std::set<std::string>();
    return conf_item_spec;
}

void conf_item_spec_free(conf_item_spec_type *conf_item_spec) {
    free(conf_item_spec->name);
    free(conf_item_spec->default_value);
    free(conf_item_spec->help);
    delete conf_item_spec->restriction;
    free(conf_item_spec);
}

void conf_item_spec_free__(void *conf_item_spec) {
    conf_item_spec_free((conf_item_spec_type *)conf_item_spec);
}

conf_item_type *conf_item_alloc(const conf_item_spec_type *conf_item_spec,
                                const char *value) {
    conf_item_type *conf_item =
        (conf_item_type *)util_malloc(sizeof *conf_item);

    assert(conf_item_spec != NULL);
    assert(value != NULL);

    conf_item->conf_item_spec = conf_item_spec;
    if (conf_item_spec->dt == DT_FILE)
        conf_item->value = util_alloc_abs_path(value);
    else
        conf_item->value = util_alloc_string_copy(value);

    return conf_item;
}

void conf_item_free(conf_item_type *conf_item) {
    free(conf_item->value);
    free(conf_item);
}

void conf_item_free__(void *conf_item) {
    conf_item_free((conf_item_type *)conf_item);
}

static conf_item_mutex_type *
conf_item_mutex_alloc(const conf_class_type *super_class, bool require_one,
                      bool inverse) {
    conf_item_mutex_type *conf_item_mutex =
        (conf_item_mutex_type *)util_malloc(sizeof *conf_item_mutex);

    conf_item_mutex->super_class = super_class;
    conf_item_mutex->require_one = require_one;
    conf_item_mutex->inverse = inverse;
    conf_item_mutex->item_spec_refs = hash_alloc();

    return conf_item_mutex;
}

void conf_item_mutex_free(conf_item_mutex_type *conf_item_mutex) {
    hash_free(conf_item_mutex->item_spec_refs);
    free(conf_item_mutex);
}

void conf_item_mutex_free__(void *conf_item_mutex) {
    conf_item_mutex_free((conf_item_mutex_type *)conf_item_mutex);
}

static bool conf_class_has_super_class(const conf_class_type *conf_class,
                                       const conf_class_type *super_class) {
    assert(conf_class != NULL);
    assert(super_class != NULL);

    const conf_class_type *parent = conf_class->super_class;

    while (parent != NULL) {
        if (parent == super_class)
            return true;
        else
            parent = parent->super_class;
    }
    return false;
}

void conf_class_insert_owned_sub_class(conf_class_type *conf_class,
                                       conf_class_type *sub_conf_class) {
    assert(conf_class != NULL);
    assert(sub_conf_class != NULL);

    /* Abort if conf_class already has an item with the same name. */
    if (hash_has_key(conf_class->item_specs, sub_conf_class->class_name))
        util_abort("%s: Internal error. conf class already has an item with "
                   "name \"%s\".\n",
                   __func__, sub_conf_class->class_name);

    /* Abort if sub_conf_class is equal to conf_class. */
    if (sub_conf_class == conf_class)
        util_abort("%s: Internal error. Trying to make a class it's own super "
                   "class.\n",
                   __func__);

    /* Abort if sub_conf_class is a super class to conf_class. */
    if (conf_class_has_super_class(conf_class, sub_conf_class))
        util_abort("%s: Internal error. Trying to make a class it's own super "
                   "class .\n",
                   __func__);

    /* Abort if sub_conf_class already has a super class. */
    if (sub_conf_class->super_class != NULL)
        util_abort(
            "%s: Internal error. Inserted class already has a super class.\n",
            __func__);

    hash_insert_hash_owned_ref(conf_class->sub_classes,
                               sub_conf_class->class_name, sub_conf_class,
                               conf_class_free__);

    sub_conf_class->super_class = conf_class;
}

void conf_class_insert_owned_item_spec(conf_class_type *conf_class,
                                       conf_item_spec_type *item_spec) {
    assert(conf_class != NULL);
    assert(item_spec != NULL);

    /* Abort if item_spec already has a super class. */
    if (item_spec->super_class != NULL)
        util_abort(
            "%s: Internal error: item is already assigned to another class.\n",
            __func__);

    /* Abort if the class has a sub class with the same name.. */
    if (hash_has_key(conf_class->sub_classes, item_spec->name))
        util_abort("%s: Internal error. conf class already has a sub class "
                   "with name \"%s\".\n",
                   __func__, item_spec->name);

    hash_insert_hash_owned_ref(conf_class->item_specs, item_spec->name,
                               item_spec, conf_item_spec_free__);

    item_spec->super_class = conf_class;
}

conf_item_mutex_type *conf_class_new_item_mutex(conf_class_type *conf_class,
                                                bool require_one,
                                                bool inverse) {
    assert(conf_class != NULL);
    conf_item_mutex_type *mutex =
        conf_item_mutex_alloc(conf_class, require_one, inverse);
    vector_append_owned_ref(conf_class->item_mutexes, mutex,
                            conf_item_mutex_free__);
    return mutex;
}

void conf_instance_insert_owned_sub_instance(
    conf_instance_type *conf_instance, conf_instance_type *sub_conf_instance) {
    assert(conf_instance != NULL);
    assert(sub_conf_instance != NULL);

    /* Abort if the instance is of unknown type. */
    if (sub_conf_instance->conf_class->super_class != conf_instance->conf_class)
        util_abort(
            "%s: Internal error. Trying to insert instance of unknown type.\n",
            __func__);

    /* Check if the instance's class is singleton. If so, remove the old instance. */
    if (sub_conf_instance->conf_class->singleton) {
        stringlist_type *instances =
            conf_instance_alloc_list_of_sub_instances_of_class(
                conf_instance, sub_conf_instance->conf_class);
        int num_instances = stringlist_get_size(instances);

        for (int i = 0; i < num_instances; i++) {
            const char *key = stringlist_iget(instances, i);
            printf("WARNING: Class \"%s\" is of singleton type. Overwriting "
                   "instance \"%s\" with \"%s\".\n",
                   sub_conf_instance->conf_class->class_name, key,
                   sub_conf_instance->name);
            hash_del(conf_instance->sub_instances, key);
        }

        stringlist_free(instances);
    }

    /* Warn if the sub_instance already exists and is overwritten. */
    if (hash_has_key(conf_instance->sub_instances, sub_conf_instance->name)) {
        printf("WARNING: Overwriting instance \"%s\" of class \"%s\" in "
               "instance \"%s\" of class \"%s\"\n",
               sub_conf_instance->name,
               conf_instance_get_class_name_ref(sub_conf_instance),
               conf_instance->name,
               conf_instance_get_class_name_ref(conf_instance));
    }

    hash_insert_hash_owned_ref(conf_instance->sub_instances,
                               sub_conf_instance->name, sub_conf_instance,
                               conf_instance_free__);
}

void conf_instance_insert_owned_item(conf_instance_type *conf_instance,
                                     conf_item_type *conf_item) {
    assert(conf_instance != NULL);
    assert(conf_item != NULL);

    const char *item_name = conf_item->conf_item_spec->name;

    /* Check that the inserted item is of known type. */
    {
        const hash_type *item_spec_hash = conf_instance->conf_class->item_specs;
        const conf_item_spec_type *conf_item_spec =
            (const conf_item_spec_type *)hash_get(item_spec_hash, item_name);
        if (conf_item_spec != conf_item->conf_item_spec)
            util_abort("%s: Internal error.\n", __func__);
    }

    hash_insert_hash_owned_ref(conf_instance->items, item_name, conf_item,
                               conf_item_free__);
}

void conf_instance_insert_item(conf_instance_type *conf_instance,
                               const char *item_name, const char *value) {
    assert(conf_instance != NULL);
    assert(item_name != NULL);
    assert(value != NULL);

    conf_item_type *conf_item;
    const conf_class_type *conf_class = conf_instance->conf_class;
    const conf_item_spec_type *conf_item_spec;

    if (!conf_class_has_item_spec(conf_class, item_name))
        util_abort("%s: Internal error. Unkown item \"%s\" in class \"%s\".\n",
                   __func__, item_name, conf_instance->conf_class->class_name);

    conf_item_spec = conf_class_get_item_spec_ref(conf_class, item_name);

    conf_item = conf_item_alloc(conf_item_spec, value);
    conf_instance_insert_owned_item(conf_instance, conf_item);
}

void conf_item_mutex_add_item_spec(conf_item_mutex_type *conf_item_mutex,
                                   const conf_item_spec_type *conf_item_spec) {

    if (conf_item_mutex->super_class != NULL) {
        const conf_class_type *conf_class = conf_item_mutex->super_class;
        const char *item_key = conf_item_spec->name;

        if (!hash_has_key(conf_class->item_specs, item_key)) {
            util_abort("%s: Internal error. Trying to insert a mutex on item "
                       "\"%s\", which class \"%s\" does not have.\n",
                       __func__, item_key, conf_class->class_name);
        } else {
            const conf_item_spec_type *conf_item_spec_class =
                (const conf_item_spec_type *)hash_get(conf_class->item_specs,
                                                      item_key);
            if (conf_item_spec_class != conf_item_spec) {
                util_abort(
                    "Internal error. Trying to insert a mutex on item \"%s\", "
                    "which class \"%s\" has a different implementation of.\n",
                    __func__, item_key, conf_class->class_name);
            }
        }
    }

    if (conf_item_mutex->require_one && conf_item_spec->required_set)
        util_abort("%s: Trying to add item \"%s\" to a mutex, but it is "
                   "required set!\n",
                   __func__, conf_item_spec->name);

    hash_insert_ref(conf_item_mutex->item_spec_refs, conf_item_spec->name,
                    conf_item_spec);
}

void conf_class_set_help(conf_class_type *conf_class, const char *help) {
    conf_class->help = util_realloc_string_copy(conf_class->help, help);
}

void conf_item_spec_add_restriction(conf_item_spec_type *conf_item_spec,
                                    const char *restriction) {
    conf_item_spec->restriction->insert(restriction);
}

void conf_item_spec_set_default_value(conf_item_spec_type *conf_item_spec,
                                      const char *default_value) {
    if (conf_item_spec->default_value != NULL)
        free(conf_item_spec->default_value);
    if (default_value != NULL)
        conf_item_spec->default_value = util_alloc_string_copy(default_value);
    else
        conf_item_spec->default_value = NULL;
}

bool conf_class_has_item_spec(const conf_class_type *conf_class,
                              const char *item_name) {
    if (!hash_has_key(conf_class->item_specs, item_name))
        return false;
    else
        return true;
}

bool conf_class_has_sub_class(const conf_class_type *conf_class,
                              const char *sub_class_name) {
    if (!hash_has_key(conf_class->sub_classes, sub_class_name))
        return false;
    else
        return true;
}

const conf_item_spec_type *
conf_class_get_item_spec_ref(const conf_class_type *conf_class,
                             const char *item_name) {
    if (!hash_has_key(conf_class->item_specs, item_name))
        util_abort("%s: Internal error.\n", __func__);

    return (const conf_item_spec_type *)hash_get(conf_class->item_specs,
                                                 item_name);
}

const conf_class_type *
conf_class_get_sub_class_ref(const conf_class_type *conf_class,
                             const char *sub_class_name) {
    if (!hash_has_key(conf_class->sub_classes, sub_class_name))
        util_abort("%s: Internal error.\n", __func__);

    return (const conf_class_type *)hash_get(conf_class->sub_classes,
                                             sub_class_name);
}

const char *
conf_instance_get_name_ref(const conf_instance_type *conf_instance) {
    return conf_instance->name;
}

bool conf_instance_is_of_class(const conf_instance_type *conf_instance,
                               const char *class_name) {
    if (strcmp(conf_instance->conf_class->class_name, class_name) == 0)
        return true;
    else
        return false;
}

bool conf_instance_has_item(const conf_instance_type *conf_instance,
                            const char *item_name) {
    if (!hash_has_key(conf_instance->items, item_name))
        return false;
    else
        return true;
}

const conf_instance_type *
conf_instance_get_sub_instance_ref(const conf_instance_type *conf_instance,
                                   const char *sub_instance_name) {
    if (!hash_has_key(conf_instance->sub_instances, sub_instance_name)) {
        util_abort("%s: Instance %s of type %s has no sub instance named %s.\n",
                   __func__, conf_instance->name,
                   conf_instance->conf_class->class_name, sub_instance_name);
    }
    return (const conf_instance_type *)hash_get(conf_instance->sub_instances,
                                                sub_instance_name);
}

stringlist_type *conf_instance_alloc_list_of_sub_instances_of_class(
    const conf_instance_type *conf_instance,
    const conf_class_type *conf_class) {
    stringlist_type *instances = stringlist_alloc_new();
    int num_sub_instances = hash_get_size(conf_instance->sub_instances);
    char **sub_instance_keys = hash_alloc_keylist(conf_instance->sub_instances);

    for (int key_nr = 0; key_nr < num_sub_instances; key_nr++) {
        const conf_instance_type *sub_instance =
            (const conf_instance_type *)hash_get(conf_instance->sub_instances,
                                                 sub_instance_keys[key_nr]);

        const conf_class_type *sub_instance_class = sub_instance->conf_class;

        if (sub_instance_class == conf_class)
            stringlist_append_copy(instances, sub_instance_keys[key_nr]);
    }

    util_free_stringlist(sub_instance_keys, num_sub_instances);

    return instances;
}

stringlist_type *conf_instance_alloc_list_of_sub_instances_of_class_by_name(
    const conf_instance_type *conf_instance, const char *sub_class_name) {
    if (!conf_class_has_sub_class(conf_instance->conf_class, sub_class_name))
        util_abort("%s: Instance \"%s\" is of class \"%s\" which has no sub "
                   "class with name \"%s\"\n",
                   conf_instance->name, conf_instance->conf_class->class_name,
                   sub_class_name);

    const conf_class_type *conf_class =
        conf_class_get_sub_class_ref(conf_instance->conf_class, sub_class_name);

    return conf_instance_alloc_list_of_sub_instances_of_class(conf_instance,
                                                              conf_class);
}

const char *
conf_instance_get_class_name_ref(const conf_instance_type *conf_instance) {
    return conf_instance->conf_class->class_name;
}

const char *
conf_instance_get_item_value_ref(const conf_instance_type *conf_instance,
                                 const char *item_name) {
    if (!hash_has_key(conf_instance->items, item_name)) {
        util_abort("%s: Instance %s of type %s has no item %s.\n", __func__,
                   conf_instance->name, conf_instance->conf_class->class_name,
                   item_name);
    }
    const conf_item_type *conf_item =
        (const conf_item_type *)hash_get(conf_instance->items, item_name);
    return conf_item->value;
}

/** If the dt supports it, this function shall return the item value as an int.
    If the dt does not support it, the function will abort.
*/
int conf_instance_get_item_value_int(const conf_instance_type *conf_instance,
                                     const char *item_name) {
    if (!hash_has_key(conf_instance->items, item_name))
        util_abort("%s: Instance %s of type %s has no item %s.\n", __func__,
                   conf_instance->name, conf_instance->conf_class->class_name,
                   item_name);

    const conf_item_type *conf_item =
        (const conf_item_type *)hash_get(conf_instance->items, item_name);
    const conf_item_spec_type *conf_item_spec = conf_item->conf_item_spec;

    return conf_data_get_int_from_string(conf_item_spec->dt, conf_item->value);
}

/** If the dt supports it, this function shall return the item value as a double.
    If the dt does not support it, the function will abort.
*/
double
conf_instance_get_item_value_double(const conf_instance_type *conf_instance,
                                    const char *item_name) {
    if (!hash_has_key(conf_instance->items, item_name))
        util_abort("%s: Instance %s of type %s has no item %s.\n", __func__,
                   conf_instance->name, conf_instance->conf_class->class_name,
                   item_name);

    const conf_item_type *conf_item =
        (const conf_item_type *)hash_get(conf_instance->items, item_name);
    const conf_item_spec_type *conf_item_spec =
        (const conf_item_spec_type *)conf_item->conf_item_spec;

    return conf_data_get_double_from_string(conf_item_spec->dt,
                                            conf_item->value);
}

/** If the dt supports it, this function shall return the item value as a time_t.
    If the dt does not support it, the function will abort.
*/
time_t
conf_instance_get_item_value_time_t(const conf_instance_type *conf_instance,
                                    const char *item_name) {
    if (!hash_has_key(conf_instance->items, item_name))
        util_abort("%s: Instance %s of type %s has no item %s.\n", __func__,
                   conf_instance->name, conf_instance->conf_class->class_name,
                   item_name);

    const conf_item_type *conf_item =
        (const conf_item_type *)hash_get(conf_instance->items, item_name);
    const conf_item_spec_type *conf_item_spec = conf_item->conf_item_spec;

    return conf_data_get_time_t_from_string(conf_item_spec->dt,
                                            conf_item->value);
}

static void
conf_item_spec_printf_help(const conf_item_spec_type *conf_item_spec) {
    assert(conf_item_spec->super_class != NULL);
    int num_restrictions = conf_item_spec->restriction->size();

    printf("\n       Help on item \"%s\" in class \"%s\":\n\n",
           conf_item_spec->name, conf_item_spec->super_class->class_name);
    printf("       - Data type    : %s\n\n",
           conf_data_get_dt_name_ref(conf_item_spec->dt));
    if (conf_item_spec->default_value != NULL)
        printf("       - Default value: %s\n\n", conf_item_spec->default_value);
    if (conf_item_spec->help != NULL)
        printf("       - %s\n", conf_item_spec->help);

    if (num_restrictions > 0) {
        printf("\n       The item \"%s\" is restricted to the following "
               "values:\n\n",
               conf_item_spec->name);
        int i = 0;
        for (auto iter = conf_item_spec->restriction->begin();
             iter != conf_item_spec->restriction->end(); ++iter, ++i)
            printf("    %i.  %s\n", i + 1, iter->c_str());
    }
    printf("\n");
}

static void conf_class_printf_help(const conf_class_type *conf_class) {
    /* TODO Should print info on the required sub classes and items. */

    if (conf_class->help != NULL) {
        if (conf_class->super_class != NULL)
            printf("\n       Help on class \"%s\" with super class \"%s\":\n\n",
                   conf_class->class_name, conf_class->super_class->class_name);
        else
            printf("\n       Help on class \"%s\":\n\n",
                   conf_class->class_name);

        printf("       %s\n", conf_class->help);
    }
    printf("\n");
}

static bool conf_item_validate(const conf_item_type *conf_item) {
    assert(conf_item != NULL);

    bool ok = true;
    const conf_item_spec_type *conf_item_spec = conf_item->conf_item_spec;
    int num_restrictions = conf_item_spec->restriction->size();
    ;

    if (!conf_data_validate_string_as_dt_value(conf_item_spec->dt,
                                               conf_item->value)) {
        ok = false;
        printf("ERROR: Failed to validate \"%s\" as a %s for item \"%s\".\n",
               conf_item->value, conf_data_get_dt_name_ref(conf_item_spec->dt),
               conf_item_spec->name);
    }

    if (num_restrictions > 0 && ok) {
        std::vector<std::string> restriction_keys;
        for (auto iter = conf_item_spec->restriction->begin();
             iter != conf_item_spec->restriction->end(); ++iter)
            restriction_keys.push_back(*iter);

        /* Legacy work-around when removing the vector supprt. */
        const int num_tokens = 1;
        const char **tokens = (const char **)&conf_item->value;

        for (int token_nr = 0; token_nr < num_tokens; token_nr++) {
            bool valid = false;

            for (int key_nr = 0; key_nr < num_restrictions; key_nr++) {
                if (strcmp(tokens[token_nr],
                           restriction_keys[key_nr].c_str()) == 0)
                    valid = true;
            }

            if (valid == false) {
                ok = false;
                printf("ERROR: Failed to validate \"%s\" as a valid value for "
                       "item \"%s\".\n",
                       conf_item->value, conf_item_spec->name);
            }
        }
    }

    if (!ok)
        conf_item_spec_printf_help(conf_item_spec);

    return ok;
}

static bool
conf_instance_has_required_items(const conf_instance_type *conf_instance) {
    bool ok = true;
    const conf_class_type *conf_class = conf_instance->conf_class;

    int num_item_specs = hash_get_size(conf_class->item_specs);
    char **item_spec_keys = hash_alloc_keylist(conf_class->item_specs);

    for (int item_spec_nr = 0; item_spec_nr < num_item_specs; item_spec_nr++) {
        const char *item_spec_name = item_spec_keys[item_spec_nr];
        const conf_item_spec_type *conf_item_spec =
            (const conf_item_spec_type *)hash_get(conf_class->item_specs,
                                                  item_spec_name);
        if (conf_item_spec->required_set) {
            if (!hash_has_key(conf_instance->items, item_spec_name)) {
                ok = false;
                printf("ERROR: Missing item \"%s\" in instance \"%s\" of class "
                       "\"%s\"\n",
                       item_spec_name, conf_instance->name,
                       conf_instance->conf_class->class_name);
                conf_item_spec_printf_help(conf_item_spec);
            }
        }
    }

    util_free_stringlist(item_spec_keys, num_item_specs);

    return ok;
}

static bool
conf_instance_has_valid_items(const conf_instance_type *conf_instance) {
    bool ok = true;

    int num_items = hash_get_size(conf_instance->items);
    char **item_keys = hash_alloc_keylist(conf_instance->items);

    for (int item_nr = 0; item_nr < num_items; item_nr++) {
        const conf_item_type *conf_item = (const conf_item_type *)hash_get(
            conf_instance->items, item_keys[item_nr]);
        if (!conf_item_validate(conf_item))
            ok = false;
    }

    util_free_stringlist(item_keys, num_items);

    return ok;
}

static bool
conf_instance_check_item_mutex(const conf_instance_type *conf_instance,
                               const conf_item_mutex_type *conf_item_mutex) {
    std::set<std::string> items_set;
    bool ok = true;
    int num_items_set = 0;
    int num_items = hash_get_size(conf_item_mutex->item_spec_refs);
    char **item_keys = hash_alloc_keylist(conf_item_mutex->item_spec_refs);

    for (int item_nr = 0; item_nr < num_items; item_nr++) {
        const char *item_key = item_keys[item_nr];
        if (conf_instance_has_item(conf_instance, item_key)) {
            items_set.insert(item_key);
        }
    }

    num_items_set = items_set.size();

    if (conf_item_mutex->inverse) {
        /* This is an inverse mutex - all (or none) items should be set. */
        if (!((num_items_set == 0) || (num_items_set == num_items))) {
            std::vector<std::string> items_set_keys;
            ok = false;
            for (const auto &key : items_set)
                items_set_keys.push_back(key);

            printf("ERROR: Failed to validate mutal inclusion in instance "
                   "\"%s\" of class \"%s\".\n\n",
                   conf_instance->name, conf_instance->conf_class->class_name);
            printf("       When using one or more of the following items, all "
                   "must be set:\n");

            for (int item_nr = 0; item_nr < num_items; item_nr++)
                printf("       %i : %s\n", item_nr, item_keys[item_nr]);

            printf("\n");
            printf("       However, only the following items were set:\n");

            for (int item_nr = 0; item_nr < num_items_set; item_nr++)
                printf("       %i : %s\n", item_nr,
                       items_set_keys[item_nr].c_str());

            printf("\n");
        }
    } else {
        if (num_items_set > 1) {
            std::vector<std::string> items_set_keys;
            ok = false;
            for (const auto &key : items_set)
                items_set_keys.push_back(key);

            printf("ERROR: Failed to validate mutal inclusion in instance "
                   "\"%s\" of class \"%s\".\n\n",
                   conf_instance->name, conf_instance->conf_class->class_name);
            printf("ERROR: Failed to validate mutex in instance \"%s\" of "
                   "class \"%s\".\n\n",
                   conf_instance->name, conf_instance->conf_class->class_name);
            printf("       Only one of the following items may be set:\n");
            for (int item_nr = 0; item_nr < num_items; item_nr++)
                printf("       %i : %s\n", item_nr, item_keys[item_nr]);

            printf("\n");
            printf("       However, all the following items were set:\n");
            for (int item_nr = 0; item_nr < num_items_set; item_nr++)
                printf("       %i : %s\n", item_nr,
                       items_set_keys[item_nr].c_str());
            printf("\n");
        }
    }

    if (num_items_set == 0 && conf_item_mutex->require_one && num_items > 0) {
        ok = false;
        printf("ERROR: Failed to validate mutex in instance \"%s\" of class "
               "\"%s\".\n\n",
               conf_instance->name, conf_instance->conf_class->class_name);
        printf("       One of the following items MUST be set:\n");
        for (int item_nr = 0; item_nr < num_items; item_nr++)
            printf("       %i : %s\n", item_nr, item_keys[item_nr]);
        printf("\n");
    }

    util_free_stringlist(item_keys, num_items);
    return ok;
}

static bool
conf_instance_has_valid_mutexes(const conf_instance_type *conf_instance) {
    bool ok = true;
    const conf_class_type *conf_class = conf_instance->conf_class;
    const vector_type *item_mutexes = conf_class->item_mutexes;
    int num_mutexes = vector_get_size(item_mutexes);

    for (int mutex_nr = 0; mutex_nr < num_mutexes; mutex_nr++) {
        const conf_item_mutex_type *conf_item_mutex =
            (const conf_item_mutex_type *)vector_iget(item_mutexes, mutex_nr);
        if (!conf_instance_check_item_mutex(conf_instance, conf_item_mutex))
            ok = false;
    }

    return ok;
}

static bool
__instance_has_sub_instance_of_type(const conf_class_type *__conf_class,
                                    int num_sub_instances,
                                    conf_class_type **class_signatures) {
    for (int sub_instance_nr = 0; sub_instance_nr < num_sub_instances;
         sub_instance_nr++) {
        if (class_signatures[sub_instance_nr] == __conf_class)
            return true;
    }
    return false;
}

static bool conf_instance_has_required_sub_instances(
    const conf_instance_type *conf_instance) {
    bool ok = true;

    /* This first part is just concerned with creating the function __instance_has_sub_instance_of_type. */
    int num_sub_instances = hash_get_size(conf_instance->sub_instances);
    conf_class_type **class_signatures = (conf_class_type **)util_calloc(
        num_sub_instances, sizeof *class_signatures);
    {
        char **sub_instance_keys =
            hash_alloc_keylist(conf_instance->sub_instances);
        for (int sub_instance_nr = 0; sub_instance_nr < num_sub_instances;
             sub_instance_nr++) {
            const char *sub_instance_name = sub_instance_keys[sub_instance_nr];
            const conf_instance_type *sub_conf_instance =
                (const conf_instance_type *)hash_get(
                    conf_instance->sub_instances, sub_instance_name);
            class_signatures[sub_instance_nr] =
                (conf_class_type *)sub_conf_instance->conf_class;
        }
        util_free_stringlist(sub_instance_keys, num_sub_instances);
    }

    /* OK, we now check that the sub classes that have require_instance true have at least one instance. */
    {
        const conf_class_type *conf_class = conf_instance->conf_class;
        int num_sub_classes = hash_get_size(conf_class->sub_classes);
        char **sub_class_keys = hash_alloc_keylist(conf_class->sub_classes);

        for (int sub_class_nr = 0; sub_class_nr < num_sub_classes;
             sub_class_nr++) {
            const char *sub_class_name = sub_class_keys[sub_class_nr];
            const conf_class_type *sub_conf_class =
                (const conf_class_type *)hash_get(conf_class->sub_classes,
                                                  sub_class_name);
            if (sub_conf_class->require_instance) {
                if (!__instance_has_sub_instance_of_type(
                        sub_conf_class, num_sub_instances, class_signatures)) {
                    printf("ERROR: Missing required instance of sub class "
                           "\"%s\" in instance \"%s\" of class \"%s\".\n",
                           sub_conf_class->class_name, conf_instance->name,
                           conf_instance->conf_class->class_name);
                    conf_class_printf_help(sub_conf_class);
                    ok = false;
                }
            }
        }

        util_free_stringlist(sub_class_keys, num_sub_classes);
    }

    free(class_signatures);

    return ok;
}

static bool
conf_instance_validate_sub_instances(const conf_instance_type *conf_instance) {
    bool ok = true;

    int num_sub_instances = hash_get_size(conf_instance->sub_instances);
    char **sub_instance_keys = hash_alloc_keylist(conf_instance->sub_instances);

    for (int sub_instance_nr = 0; sub_instance_nr < num_sub_instances;
         sub_instance_nr++) {
        const char *sub_instances_key = sub_instance_keys[sub_instance_nr];
        const conf_instance_type *sub_conf_instance =
            (const conf_instance_type *)hash_get(conf_instance->sub_instances,
                                                 sub_instances_key);
        if (!conf_instance_validate(sub_conf_instance))
            ok = false;
    }

    util_free_stringlist(sub_instance_keys, num_sub_instances);

    return ok;
}

/**
 * This function resolves paths for each file-keyword in the given
 * configuration. It returns a set of keywords with corresponding
 * path for each non-existing path.
 *
 * The return type is std::set because this is a simple and efficient
 * way to avoid duplicates in the end-result: Note that the method
 * recurses and combines results in the last part of the code.
 *
 * Elements in the set are the keyword concatenated with "=>" and its
 * resolved path. For example
 *
 *     "OBS_FILE=>/var/tmp/obs_path/obs.txt"
 *
 */
static std::set<std::string>
get_path_errors(const conf_instance_type *conf_instance) {
    std::set<std::string> path_errors;
    {
        int num_items = hash_get_size(conf_instance->items);
        char **item_keys = hash_alloc_keylist(conf_instance->items);

        for (int item_nr = 0; item_nr < num_items; item_nr++) {
            const conf_item_type *conf_item = (const conf_item_type *)hash_get(
                conf_instance->items, item_keys[item_nr]);
            const conf_item_spec_type *conf_item_spec =
                conf_item->conf_item_spec;
            if (conf_item_spec->dt == DT_FILE) {
                if (!fs::exists(conf_item->value))
                    path_errors.insert(std::string(conf_item_spec->name) +
                                       "=>" + std::string(conf_item->value));
            }
        }
        util_free_stringlist(item_keys, num_items);
    }

    {
        int num_sub_instances = hash_get_size(conf_instance->sub_instances);
        char **sub_instance_keys =
            hash_alloc_keylist(conf_instance->sub_instances);

        for (int sub_instance_nr = 0; sub_instance_nr < num_sub_instances;
             sub_instance_nr++) {
            const char *sub_instances_key = sub_instance_keys[sub_instance_nr];
            const conf_instance_type *sub_conf_instance =
                (const conf_instance_type *)hash_get(
                    conf_instance->sub_instances, sub_instances_key);
            std::set<std::string> sub = get_path_errors(sub_conf_instance);
            path_errors.insert(sub.begin(), sub.end());
        }

        util_free_stringlist(sub_instance_keys, num_sub_instances);
    }

    return path_errors;
}

/**
 * This method returns a single C-style string (zero-terminated char *)
 * describing keywords and paths to which these resolve, and which does
 * not exist. For example
 *
 * "OBS_FILE=>/var/tmp/obs_path/obs.txt\nINDEX_FILE=>/var/notexisting/file.txt"
 *
 * Note newlines - this string is intended for printing.
 */
char *conf_instance_get_path_error(const conf_instance_type *conf_instance) {
    std::set<std::string> errors = get_path_errors(conf_instance);

    if (errors.size() == 0)
        return NULL;

    std::string retval;
    for (std::string s : errors) {
        retval.append(s);
        retval.append("\n");
    }
    return strdup(retval.data());
}

bool conf_instance_validate(const conf_instance_type *conf_instance) {
    return conf_instance && conf_instance_has_required_items(conf_instance) &&
           conf_instance_has_valid_mutexes(conf_instance) &&
           conf_instance_has_valid_items(conf_instance) &&
           conf_instance_has_required_sub_instances(conf_instance) &&
           conf_instance_validate_sub_instances(conf_instance);
}

static void conf_instance_parser_add_item(conf_instance_type *conf_instance,
                                          const char *item_name,
                                          char **buffer_pos) {
    char *token_assign;
    char *token_value;
    char *token_end;

    char *buffer_pos_loc = *buffer_pos;

    token_assign = conf_util_alloc_next_token(&buffer_pos_loc);
    if (token_assign == NULL) {
        /* This will fail. Give up. */
        printf(
            "WARNING: Unexpected EOF after \"%s\". Giving up on this item.\n\n",
            item_name);
        return;
    } else if (strcmp(token_assign, "=") != 0) {
        /* This will fail. Give up. */
        printf("WARNING: Unexpected \"%s\" after \"%s\". Giving up on this "
               "item.\n\n",
               token_assign, item_name);
        free(token_assign);
        *buffer_pos = buffer_pos_loc;
        return;
    }

    token_value = conf_util_alloc_next_token(&buffer_pos_loc);
    if (token_value == NULL) {
        /* This will fail. Give up. */
        printf("WARNING: Unexpected EOF after \"%s = \". Giving up on this "
               "item.\n\n",
               item_name);
        free(token_assign);
        return;
    } else
        conf_instance_insert_item(conf_instance, item_name, token_value);

    *buffer_pos = buffer_pos_loc;

    token_end = conf_util_alloc_next_token(&buffer_pos_loc);
    if (token_end == NULL) {
        /* We've already alloc'd the token. Print a warning to the user. */
        printf("WARNING: Unexpected EOF after \"%s = %s \".\n\n", item_name,
               token_value);
        free(token_assign);
        free(token_value);
        return;
    } else if (strcmp(token_end, ";") != 0) {
        printf("WARNING: Unexpected \"%s\" after \"%s = %s \". Probably a "
               "missing \";\".\n\n",
               token_end, item_name, token_value);
    } else {
        *buffer_pos = buffer_pos_loc;
    }

    free(token_assign);
    free(token_value);
    free(token_end);
}

static void conf_instance_parser_skip_unknown_class(char **buffer_pos) {
    int depth_in_unkown_class = 1;
    char *token = conf_util_alloc_next_token(buffer_pos);

    while (token != NULL) {
        if (strcmp(token, "{") == 0)
            depth_in_unkown_class++;
        else if (strcmp(token, "}") == 0)
            depth_in_unkown_class--;

        printf("WARNING: Skipping token \"%s\" in unknown class.\n", token);
        free(token);
        if (depth_in_unkown_class == 0)
            break;
        else
            token = conf_util_alloc_next_token(buffer_pos);
    }
}

static void conf_instance_add_data_from_token_buffer(
    conf_instance_type *conf_instance, char **buffer_pos, bool allow_inclusion,
    bool is_root, const char *current_file_name) {
    const conf_class_type *conf_class = conf_instance->conf_class;
    char *token = conf_util_alloc_next_token(buffer_pos);

    bool scope_start_set = false;
    bool scope_end_set = false;

    while (token != NULL) {
        if (conf_class_has_item_spec(conf_class, token) &&
            (scope_start_set || is_root))
            conf_instance_parser_add_item(conf_instance, token, buffer_pos);
        else if (conf_class_has_sub_class(conf_class, token) &&
                 (scope_start_set || is_root)) {
            char *name = conf_util_alloc_next_token(buffer_pos);
            const conf_class_type *sub_conf_class =
                conf_class_get_sub_class_ref(conf_class, token);
            if (name != NULL) {
                conf_instance_type *sub_conf_instance =
                    conf_instance_alloc_default(sub_conf_class, name);
                free(name);
                conf_instance_insert_owned_sub_instance(conf_instance,
                                                        sub_conf_instance);
                conf_instance_add_data_from_token_buffer(
                    sub_conf_instance, buffer_pos, allow_inclusion, false,
                    current_file_name);
            } else
                printf("WARNING: Unexpected EOF after \"%s\" in file %s.\n\n",
                       token, current_file_name);
        } else if (strcmp(token, "}") == 0) {
            if (scope_start_set) {
                scope_end_set = true;
                free(token);
                break;
            } else
                printf("WARNING: Skipping unexpected token \"%s\" in file "
                       "%s.\n\n",
                       token, current_file_name);
        } else if (strcmp(token, "{") == 0) {
            if (!scope_start_set && !is_root)
                scope_start_set = true;
            else
                conf_instance_parser_skip_unknown_class(buffer_pos);
        } else if (strcmp(token, ";") == 0) {
            if (!scope_start_set) {
                free(token);
                break;
            } else
                printf("WARNING: Skipping unexpected token \"%s\" in file "
                       "%s.\n\n",
                       token, current_file_name);
        } else if (strcmp(token, "include") == 0) {
            char *file_name =
                util_alloc_abs_path(conf_util_alloc_next_token(buffer_pos));
            char *buffer_pos_lookahead = *buffer_pos;
            char *token_end;

            if (file_name == NULL) {
                printf("WARNING: Unexpected EOF after \"%s\".\n\n", token);
                free(token);
                break;
            } else if (!allow_inclusion) {
                printf("WARNING: No support for nested inclusion. Skipping "
                       "file \"%s\".\n\n",
                       file_name);
            } else {
                path_stack_type *path_stack = path_stack_alloc();
                path_stack_push_cwd(path_stack);
                util_chdir_file(file_name);
                {
                    char *buffer_new =
                        conf_util_fscanf_alloc_token_buffer(file_name);
                    char *buffer_pos_new = buffer_new;

                    conf_instance_add_data_from_token_buffer(
                        conf_instance, &buffer_pos_new, false, true, file_name);

                    free(buffer_new);
                }
                path_stack_pop(path_stack);
                path_stack_free(path_stack);
            }

            /* Check that the filename is followed by a ; */
            token_end = conf_util_alloc_next_token(&buffer_pos_lookahead);
            if (token_end == NULL) {
                printf("WARNING: Unexpected EOF after inclusion of file "
                       "\"%s\".\n\n",
                       file_name);
                free(token);
                free(file_name);
                break;
            } else if (strcmp(token_end, ";") != 0) {
                printf("WARNING: Unexpected \"%s\" after inclusion of file "
                       "\"%s\". Probably a missing \";\".\n\n",
                       token_end, file_name);
            } else {
                *buffer_pos = buffer_pos_lookahead;
            }
            free(token_end);
            free(file_name);
        } else if (strcmp(token, "BLOCK_OBSERVATION") == 0) {
            std::string msg = "The keyword BLOCK_OBSERVATION is no longer "
                              "supported. For RFT use GENDATA_RFT.\n";
            logger->error(msg);
            fprintf(stderr, "%s", msg.c_str());
            util_abort(msg.c_str());
        } else {
            printf("WARNING: Skipping unexpected token \"%s\" in file "
                   "%s.\n\n",
                   token, current_file_name);
        }

        free(token);
        token = conf_util_alloc_next_token(buffer_pos);
    }

    if (scope_end_set) {
        token = conf_util_alloc_next_token(buffer_pos);
        if (token == NULL) {
            printf("WARNING: Unexpected EOF. Missing terminating \";\" in file "
                   "%s.\n",
                   current_file_name);
        } else if (strcmp(token, ";") != 0) {
            printf("WARNING: Missing terminating \";\" at the end of \"%s\" in "
                   "file %s.\n",
                   conf_instance->name, current_file_name);
            free(token);
        } else
            free(token);
    }
}

conf_instance_type *
conf_instance_alloc_from_file(const conf_class_type *conf_class,
                              const char *name, const char *file_name) {
    conf_instance_type *conf_instance =
        conf_instance_alloc_default(conf_class, name);
    path_stack_type *path_stack = path_stack_alloc();
    char *file_arg = util_split_alloc_filename(file_name);
    path_stack_push_cwd(path_stack);

    util_chdir_file(file_name);
    {
        char *buffer = conf_util_fscanf_alloc_token_buffer(file_arg);
        char *buffer_pos = buffer;

        conf_instance_add_data_from_token_buffer(conf_instance, &buffer_pos,
                                                 true, true, file_name);

        free(buffer);
    }
    free(file_arg);
    path_stack_pop(path_stack);
    path_stack_free(path_stack);
    return conf_instance;
}
