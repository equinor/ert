#include <algorithm>
#include <filesystem>
#include <iostream>

#include <cassert>
#include <cstring>

#include <ert/logging.hpp>
#include <ert/util/path_stack.hpp>

#include <ert/config/conf.hpp>
#include <ert/config/conf_util.hpp>
#include <utility>

namespace fs = std::filesystem;

static auto logger = ert::get_logger("config");

conf_instance::conf_instance(std::shared_ptr<conf_class> cls, std::string name)
    : cls(cls), name(std::move(name)), sub_instances(), items() {
    /* Insert items that have a default value in their specs. */
    for (auto &[item_name, spec] : cls->item_specs) {
        if (spec->default_value.has_value()) {
            auto item =
                std::make_shared<conf_item>(spec, spec->default_value.value());
            items[item_name] = item;
        }
    }
}

bool conf_class::has_super_class(std::shared_ptr<conf_class> cls) {
    auto parent = this->super_class.lock();

    while (parent != nullptr) {
        if (parent == cls)
            return true;
        else
            parent = parent->super_class.lock();
    }
    return false;
}

void conf_class::insert_sub_class(std::shared_ptr<conf_class> sub_conf_class) {
    if (item_specs.count(sub_conf_class->class_name))
        throw std::logic_error(
            fmt::format("conf class already has a subclass with name \"{}\"",
                        sub_conf_class->class_name));

    if (sub_conf_class.get() == this)
        throw std::logic_error("Cannot make a class it's own super class");

    if (this->has_super_class(sub_conf_class))
        throw std::logic_error("Cannot make a class it's own super class");

    /* Abort if sub_conf_class already has a super class. */
    if (sub_conf_class->super_class.lock())
        throw std::logic_error("Inserted class already has a super class");

    this->sub_classes[sub_conf_class->class_name] = sub_conf_class;
    sub_conf_class->super_class = shared_from_this();
}

void conf_class::insert_item_spec(std::shared_ptr<conf_item_spec> item_spec) {
    /* Abort if item_spec already has a super class. */
    if (item_spec->super_class.lock())
        throw std::logic_error("Item is already assigned to another class");

    /* Abort if the class has a sub class with the same name.. */
    if (this->sub_classes.count(item_spec->name))
        throw std::logic_error(fmt::format("conf class already has a sub class "
                                           "with name \"{}\".\n",
                                           item_spec->name));

    this->item_specs[item_spec->name] = item_spec;

    item_spec->super_class = shared_from_this();
}

std::shared_ptr<conf_item_mutex> conf_class::new_item_mutex(bool require_one,
                                                            bool inverse) {
    auto mutex = std::make_shared<conf_item_mutex>(shared_from_this(),
                                                   require_one, inverse);
    this->item_mutexes.push_back(mutex);
    return mutex;
}

std::vector<std::shared_ptr<conf_instance>>
conf_instance::get_sub_instances(std::string find_name) {
    std::vector<std::shared_ptr<conf_instance>> instances;

    for (auto &[sub_instance_name, sub_instance] : this->sub_instances) {
        auto sub_instance_class = sub_instance->cls;
        if (sub_instance->cls->class_name == find_name)
            instances.push_back(sub_instance);
    }

    return instances;
}

std::string conf_instance::get_value(std::string name) {
    return items.at(name)->value;
};

void conf_instance::insert_sub_instance(
    std::shared_ptr<conf_instance> sub_conf_instance) {
    /* Abort if the instance is of unknown type. */
    if (sub_conf_instance->cls->super_class.lock() != this->cls)
        throw std::logic_error("Trying to insert instance of unknown type");

    /* Check if the instance's class is singleton. If so, remove the old instance. */
    if (sub_conf_instance->cls->singleton) {
        std::vector<std::shared_ptr<conf_instance>> instances =
            this->get_sub_instances(sub_conf_instance->cls->class_name);
        for (auto &instance : instances) {
            const std::string key = instance->name;
            printf("WARNING: Class \"%s\" is of singleton type. Overwriting "
                   "instance \"%s\" with \"%s\".\n",
                   sub_conf_instance->cls->class_name.c_str(), key.c_str(),
                   sub_conf_instance->name.c_str());
            this->sub_instances.erase(key);
        }
    }

    /* Warn if the sub_instance already exists and is overwritten. */
    if (this->sub_instances.count(sub_conf_instance->name)) {
        printf("WARNING: Overwriting instance \"%s\" of class \"%s\" in "
               "instance \"%s\" of class \"%s\"\n",
               sub_conf_instance->name.c_str(),
               sub_conf_instance->cls->class_name.c_str(), this->name.c_str(),
               this->cls->class_name.c_str());
    }

    this->sub_instances[sub_conf_instance->name] = sub_conf_instance;
}

void conf_instance::insert_item(std::string item_name, std::string value) {
    auto conf_item_spec = cls->item_specs.at(item_name);
    if (conf_item_spec->dt == DT_FILE) {
        items[item_name] = std::make_shared<conf_item>(
            conf_item_spec, util_alloc_abs_path(value.c_str()));
    } else {
        items[item_name] = std::make_shared<conf_item>(conf_item_spec, value);
    }
}

void conf_item_mutex::add_item_spec(
    std::shared_ptr<conf_item_spec> conf_item_spec) {

    if (auto conf_class = this->super_class.lock()) {
        std::string item_key = conf_item_spec->name;

        if (!conf_class->item_specs.count(item_key)) {
            throw std::logic_error(
                fmt::format("Trying to insert a mutex on item "
                            "\"{}\", which class \"{}\" does not have",
                            item_key, conf_class->class_name));
        } else {
            auto conf_item_spec_class = conf_class->item_specs[item_key];
            if (conf_item_spec_class != conf_item_spec) {
                throw std::logic_error(fmt::format(
                    "Trying to insert a mutex on item \"{}\", "
                    "which class \"{}\" has a different implementation of.\n",
                    item_key, conf_class->class_name));
            }
        }
    }

    if (require_one && conf_item_spec->required_set)
        throw std::logic_error(
            fmt::format("Trying to add item \"{}\" to a mutex, but it is "
                        "required set",
                        conf_item_spec->name));

    item_spec_refs[conf_item_spec->name] = conf_item_spec;
}

std::string conf_item_spec::get_help() {
    std::string result;

    if (auto cls = this->super_class.lock()) {
        result +=
            fmt::format("\nHelp on item \"{}\" in class \"{}\":", this->name,
                        cls->class_name);
    } else {
        result += fmt::format("\nHelp on item \"{}\":", this->name);
    }
    result += fmt::format("\n       - Data type    : {}",
                          conf_data_get_dt_name_ref(this->dt));
    if (this->default_value.has_value())
        result += fmt::format("\n       - Default value: {}",
                              this->default_value.value());
    if (this->help.has_value())
        result += fmt::format("\n       - {}", this->help.value());

    if (this->restriction.size() > 0) {
        result += fmt::format(
            "\n       The item \"{}\" is restricted to the following "
            "values:",
            this->name);
        int i = 0;
        for (auto iter = restriction.begin(); iter != restriction.end();
             ++iter, ++i)
            result += fmt::format("\n    {}.  {}", i + 1, *iter);
    }
    return result;
}

std::string conf_class::get_help() {
    /* TODO Should print info on the required sub classes and items. */
    std::string result;
    if (this->help.has_value()) {
        if (auto super_class = this->super_class.lock())
            result +=
                fmt::format("\nHelp on class \"{}\" with super class \"{}\":",
                            this->class_name, super_class->class_name);
        else
            result +=
                fmt::format("\n       Help on class \"{}\":", this->class_name);

        result += fmt::format("\n       {}", this->help.value());
    }
    return result;
}

std::vector<std::string> conf_item::validate() {
    auto conf_item_spec = this->spec;

    std::vector<std::string> errors;
    if (!conf_data_validate_string_as_dt_value(conf_item_spec->dt,
                                               this->value.c_str())) {
        errors.push_back(fmt::format(
            "Failed to validate \"{}\" as a {} for item \"{}\". {}\n",
            this->value, conf_data_get_dt_name_ref(conf_item_spec->dt),
            conf_item_spec->name, conf_item_spec->get_help()));
    }

    if (conf_item_spec->restriction.size() > 0) {
        bool valid = false;
        for (const auto &iter : conf_item_spec->restriction) {
            if (this->value == iter)
                valid = true;
        }

        if (valid == false) {
            errors.push_back(
                fmt::format("Failed to validate \"{}\" as a valid value for "
                            "item \"{}\".\n",
                            this->value, conf_item_spec->name));
        }
    }

    return errors;
}

std::vector<std::string> conf_instance::has_required_items() {
    auto conf_class = this->cls.get();

    std::vector<std::string> errors;
    for (auto &[item_spec_name, conf_item_spec] : conf_class->item_specs) {
        if (conf_item_spec->required_set) {
            if (this->items.count(item_spec_name) == 0) {
                errors.push_back(fmt::format(
                    "Missing item \"{}\" in instance \"{}\" of class "
                    "\"{}\"\n",
                    item_spec_name, this->name, this->cls->class_name,
                    conf_item_spec->get_help()));
            }
        }
    }
    return errors;
}

std::vector<std::string> conf_instance::has_valid_items() {
    std::vector<std::string> errors;
    for (auto &[item_name, conf_item] : this->items) {
        auto item_errors = conf_item->validate();
        errors.insert(errors.end(), item_errors.begin(), item_errors.end());
    }
    return errors;
}

std::vector<std::string> conf_instance::check_item_mutex(
    std::shared_ptr<conf_item_mutex> conf_item_mutex) {
    std::set<std::string> items_set;
    std::vector<std::string> errors;

    for (auto &[item_key, item_spec] : conf_item_mutex->item_spec_refs) {
        if (items.count(item_key)) {
            items_set.insert(item_key);
        }
    }

    size_t num_items_set = items_set.size();
    size_t num_items = conf_item_mutex->item_spec_refs.size();

    if (conf_item_mutex->inverse) {
        /* This is an inverse mutex - all (or none) items should be set. */
        if (!((num_items_set == 0) || (num_items_set == num_items))) {
            std::vector<std::string> items_set_keys;
            for (const auto &key : items_set)
                items_set_keys.push_back(key);

            std::string error;
            error +=
                fmt::format("Failed to validate mutual inclusion in instance "
                            "\"{}\" of class \"{}\".\n",
                            this->name, this->cls->class_name);
            error += "       When using one or more of the following items, "
                     "all must be set:\n";

            for (auto &[item_key, item_spec] : conf_item_mutex->item_spec_refs)
                error += fmt::format("       {}\n", item_key.c_str());

            error += "       However, only the following items were set:\n";

            for (int item_nr = 0; item_nr < num_items_set; item_nr++)
                error += fmt::format("       {} : {}\n", item_nr,
                                     items_set_keys[item_nr]);
            errors.push_back(error);
        }
    } else {
        if (num_items_set > 1) {
            std::vector<std::string> items_set_keys;
            for (const auto &key : items_set)
                items_set_keys.push_back(key);

            std::string error;
            error +=
                fmt::format("Failed to validate mutex in instance \"{}\" of "
                            "class \"{}\".\n",
                            this->name, this->cls->class_name);
            error += "       Only one of the following items may be set:\n";
            for (auto &[item_key, item_spec] : conf_item_mutex->item_spec_refs)
                error += fmt::format("       {}\n", item_key);

            error += "       However, all the following items were set:\n";
            for (int item_nr = 0; item_nr < num_items_set; item_nr++)
                error += fmt::format("       {} : {}\n", item_nr,
                                     items_set_keys[item_nr]);
            errors.push_back(error);
        }
    }

    if (num_items_set == 0 && conf_item_mutex->require_one && num_items > 0) {
        std::string error;
        error +=
            fmt::format("Failed to validate mutex in instance \"{}\" of class "
                        "\"{}\".\n",
                        this->name, this->cls->class_name);
        error += "       One of the following items MUST be set:\n";
        for (auto &[item_key, item_spec] : conf_item_mutex->item_spec_refs)
            error += fmt::format("       {}\n", item_key);
        errors.push_back(error);
    }
    return errors;
}

std::vector<std::string> conf_instance::has_valid_mutexes() {
    std::vector<std::string> errors;
    for (auto &mutex : cls->item_mutexes) {
        auto instance_errors = conf_instance::check_item_mutex(mutex);
        errors.insert(errors.end(), instance_errors.begin(),
                      instance_errors.end());
    }
    return errors;
}

std::vector<std::string> conf_instance::has_required_sub_instances() {

    std::vector<std::shared_ptr<conf_class>> class_signatures;
    for (auto &[sub_instance_name, sub_conf_instance] : this->sub_instances) {
        class_signatures.push_back(sub_conf_instance->cls);
    }
    std::vector<std::string> errors;

    /* check that the sub classes that have require_instance true have at least one instance. */
    auto conf_class = this->cls;
    for (auto &[sub_class_name, sub_conf_class] : conf_class->sub_classes) {
        if (sub_conf_class->require_instance) {
            if (std::count(class_signatures.begin(), class_signatures.end(),
                           sub_conf_class) > 0) {
                errors.push_back(fmt::format(
                    "Missing required instance of sub class "
                    "\"{}\" in instance \"{}\" of class \"{}\".\n{}",
                    sub_conf_class->class_name, this->name,
                    this->cls->class_name, conf_class->get_help()));
            }
        }
    }
    return errors;
}

std::vector<std::string> conf_instance::validate_sub_instances() {
    std::vector<std::string> errors;
    for (auto &[sub_instances_key, sub_conf_instance] : this->sub_instances) {
        auto sub_errors = sub_conf_instance->validate();
        errors.insert(errors.end(), sub_errors.begin(), sub_errors.end());
    }
    return errors;
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
std::set<std::string> conf_instance::get_path_errors() {
    std::set<std::string> path_errors;
    for (auto &[conf_item_name, conf_item] : this->items) {
        auto conf_item_spec = conf_item->spec;
        if (conf_item_spec->dt == DT_FILE) {
            if (!fs::exists(conf_item->value))
                path_errors.insert(std::string(conf_item_spec->name) + "=>" +
                                   std::string(conf_item->value));
        }
    }
    for (auto &[sub_instance_key, sub_conf_instance] : this->sub_instances) {
        std::set<std::string> sub = sub_conf_instance->get_path_errors();
        path_errors.insert(sub.begin(), sub.end());
    }

    return path_errors;
}
/**
 * This method returns a single string 
 * describing keywords and paths to which these resolve, and which does
 * not exist. For example
 *
 * "OBS_FILE=>/var/tmp/obs_path/obs.txt\nINDEX_FILE=>/var/notexisting/file.txt"
 *
 * Note newlines - this string is intended for printing.
 */
std::string conf_instance::get_path_error() {
    std::set<std::string> errors = this->get_path_errors();

    if (errors.size() == 0)
        return "";

    std::string retval = "The following keywords in your configuration did "
                         "not resolve to a valid path:\n ";
    for (std::string s : errors) {
        retval.append(s);
        retval.append("\n");
    }
    return retval;
}

std::vector<std::string> conf_instance::validate() {
    std::vector<std::string> errors;
    std::string path_error = this->get_path_error();
    if (path_error != "") {
        errors.push_back(path_error);
    }
    auto required_errors = this->has_required_items();
    errors.insert(errors.end(), required_errors.begin(), required_errors.end());
    auto mutex_errors = this->has_valid_mutexes();
    errors.insert(errors.end(), mutex_errors.begin(), mutex_errors.end());
    auto item_errors = this->has_valid_items();
    errors.insert(errors.end(), item_errors.begin(), item_errors.end());
    auto required_sub_errors = this->has_required_sub_instances();
    errors.insert(errors.end(), required_sub_errors.begin(),
                  required_sub_errors.end());
    auto sub_errors = this->validate_sub_instances();
    errors.insert(errors.end(), sub_errors.begin(), sub_errors.end());
    return errors;
}

static void
conf_instance_parser_add_item(std::shared_ptr<conf_instance> conf_instance,
                              const char *item_name, char **buffer_pos) {
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
        conf_instance->insert_item(item_name, token_value);

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

void conf_instance::add_data_from_token_buffer(char **buffer_pos,
                                               bool allow_inclusion,
                                               bool is_root,
                                               std::string current_file_name) {
    std::shared_ptr<conf_class> conf_class = this->cls;
    char *token = conf_util_alloc_next_token(buffer_pos);

    bool scope_start_set = false;
    bool scope_end_set = false;

    while (token != NULL) {
        if (conf_class->item_specs.count(token) > 0 &&
            (scope_start_set || is_root))
            conf_instance_parser_add_item(shared_from_this(), token,
                                          buffer_pos);
        else if (this->cls->sub_classes.count(token) > 0 &&
                 (scope_start_set || is_root)) {
            char *name = conf_util_alloc_next_token(buffer_pos);
            auto sub_conf_class = conf_class->sub_classes.at(token);
            if (name != NULL) {
                auto sub_conf_instance =
                    std::make_shared<conf_instance>(sub_conf_class, name);
                free(name);
                this->insert_sub_instance(sub_conf_instance);
                sub_conf_instance->add_data_from_token_buffer(
                    buffer_pos, allow_inclusion, false, current_file_name);
            } else
                printf("WARNING: Unexpected EOF after \"%s\" in file %s.\n\n",
                       token, current_file_name.c_str());
        } else if (strcmp(token, "}") == 0) {
            if (scope_start_set) {
                scope_end_set = true;
                free(token);
                break;
            } else
                printf("WARNING: Skipping unexpected token \"%s\" in file "
                       "%s.\n\n",
                       token, current_file_name.c_str());
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
                       token, current_file_name.c_str());
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

                    this->add_data_from_token_buffer(&buffer_pos_new, false,
                                                     true, file_name);

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
            throw std::runtime_error(
                "The keyword BLOCK_OBSERVATION is no longer "
                "supported. For RFT use GENDATA_RFT");
        } else {
            printf("WARNING: Skipping unexpected token \"%s\" in file "
                   "%s.\n\n",
                   token, current_file_name.c_str());
        }

        free(token);
        token = conf_util_alloc_next_token(buffer_pos);
    }

    if (scope_end_set) {
        token = conf_util_alloc_next_token(buffer_pos);
        if (token == NULL) {
            printf("WARNING: Unexpected EOF. Missing terminating \";\" in file "
                   "%s.\n",
                   current_file_name.c_str());
        } else if (strcmp(token, ";") != 0) {
            printf("WARNING: Missing terminating \";\" at the end of \"%s\" in "
                   "file %s.\n",
                   this->name.c_str(), current_file_name.c_str());
            free(token);
        } else
            free(token);
    }
}

std::shared_ptr<conf_instance>
conf_instance::from_file(std::shared_ptr<conf_class> conf_class,
                         std::string name, std::string file_name) {
    auto result = std::make_shared<conf_instance>(conf_class, name);
    path_stack_type *path_stack = path_stack_alloc();
    char *file_arg = util_split_alloc_filename(file_name.c_str());
    path_stack_push_cwd(path_stack);

    util_chdir_file(file_name.c_str());
    char *buffer = conf_util_fscanf_alloc_token_buffer(file_arg);
    char *buffer_pos = buffer;

    result->add_data_from_token_buffer(&buffer_pos, true, true, file_name);

    free(buffer);
    free(file_arg);
    path_stack_pop(path_stack);
    path_stack_free(path_stack);
    return result;
}
