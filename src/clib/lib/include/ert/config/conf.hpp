#pragma once

/* libconfig: lightweight configuration parser
 *
 *
 *
 * Introduction
 *
 * This library provides a lightweight configuration parser for the
 * enkf application. The goal of the library is to provide the
 * developer with a tool for rapid specification of configuration
 * files, automatic checking of user provided configuration files
 * and typed access to configuration items.
 *
 *
 *
 * A Simple Example
 *
 * Let us consider a simple example of user provided configuration
 * file that can be used with the parser:
 *
 *
 * res_sim FrontSim2007
 * {
 *   executable = /bin/frontsim2007;
 *   version    = 2007;
 *
 *   run_host bgo179lin
 *   {
 *     hostname = some.example.com;
 *     num_jobs = 4;
 *   };
 * };
 *
 *
 * Note that the newlines are not neccessary. In the example above,
 * the user has provided an instance of the class "res_sim" with name
 * FrontSim2007. Further, the user has set the items executable and version.
 * He has also provided a instance of the sub class "run_host" with name
 * bgo179lin and allocated 4 jobs to this machine.
 *
 *
 *
 * Structure
 *
 * The system is built around four basic objects:
 *
 *  - Class definitions.
 *  - Item specifications.
 *  - Instances of classes.
 *  - Instances of item specifications, i.e. items.
 *
 * The relationship between the objects is as follows :
 *
 *  - Class:
 *    . Can have contain both classes and item specifications.
 *    . Can not contain items or class instances.
 *
 *  - Item specifications:
 *    . Can not contain any of the other objects.
 *
 *  - Instances of classes:
 *    . Can contain class instances and items.
 *
 *  - Items:
 *    . Can not contain any of the other objects.
 *
 *
 *
 * General Use
 *
 * The parser is designed to be used in the following way:
 *
 *  - The developer creates the classes and item specifications needed.
 *  - Using the library and the classes, user provided configuration
 *    files are read and validated.
 *  - If the validation fails, the developer can choose to exit.
 *  - Using the library, the devloper has typed access to all
 *    information provided by the user.
 *
 */

#include <cstdbool>

#include <ert/config/conf_data.hpp>
#include <ert/util/hash.hpp>
#include <ert/util/vector.hpp>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

struct conf_item_spec;
struct conf_item_mutex;
struct conf_item_type;
struct conf_item;

class conf_class : public std::enable_shared_from_this<conf_class> {
private:
    /** Can be NULL */
    std::weak_ptr<conf_class> super_class;
    std::string class_name;
    std::optional<std::string> help;
    bool require_instance;
    bool singleton;
    std::map<std::string, std::shared_ptr<conf_class>> sub_classes;
    std::map<std::string, std::shared_ptr<conf_item_spec>> item_specs;
    std::vector<std::shared_ptr<conf_item_mutex>> item_mutexes;

    bool has_super_class(std::shared_ptr<conf_class>);
    std::string get_help();

    friend class conf_instance;
    friend class conf_item_mutex;
    friend class conf_item_spec;

public:
    conf_class(std::string class_name, bool require_instance, bool singleton,
               std::optional<std::string> help)
        : class_name(std::move(class_name)), require_instance(require_instance),
          singleton(singleton), help(std::move(help)){};

    void insert_sub_class(std::shared_ptr<conf_class>);
    void insert_item_spec(std::shared_ptr<conf_item_spec>);
    std::shared_ptr<conf_item_mutex> new_item_mutex(bool require_one,
                                                    bool inverse);
};

class conf_instance : public std::enable_shared_from_this<conf_instance> {
private:
    std::shared_ptr<conf_class> cls;
    std::map<std::string, std::shared_ptr<conf_instance>> sub_instances;
    std::map<std::string, std::shared_ptr<conf_item>> items;

    std::vector<std::string> has_required_items();
    std::vector<std::string> has_valid_items();
    std::vector<std::string> check_item_mutex(std::shared_ptr<conf_item_mutex>);
    std::vector<std::string> has_valid_mutexes();
    std::vector<std::string> has_required_sub_instances();
    std::vector<std::string> validate_sub_instances();
    std::set<std::string> get_path_errors();
    void add_data_from_token_buffer(char **buffer_pos, bool allow_inclusion,
                                    bool is_root,
                                    std::string current_file_name);

public:
    std::string name;
    conf_instance(std::shared_ptr<conf_class> cls, std::string name);
    bool has_value(std::string name) { return items.count(name); };
    std::string get_value(std::string name);
    std::vector<std::shared_ptr<conf_instance>>
    get_sub_instances(std::string sub_class_name);
    std::vector<std::string> validate();
    void insert_item(std::string item_name, std::string value);
    void insert_sub_instance(std::shared_ptr<conf_instance>);
    std::string get_path_error();

    static std::shared_ptr<conf_instance> from_file(std::shared_ptr<conf_class>,
                                                    std::string name,
                                                    std::string file_name);
};

struct conf_item_spec {
    /** NULL if not inserted into a class */
    std::weak_ptr<conf_class> super_class;
    std::string name;
    /** Require the item to take a valid value */
    bool required_set;
    std::optional<std::string> default_value;
    /** Data type. See conf_data */
    dt_enum dt;
    std::set<std::string> restriction;
    std::optional<std::string> help;
    conf_item_spec(std::string name, bool required_set, dt_enum dt,
                   std::string help)
        : name(std::move(name)), required_set(required_set), dt(dt), help(help),
          restriction(), default_value(), super_class(){};
    void add_restriction(std::string restriction) {
        this->restriction.insert(restriction);
    };
    std::string get_help();
};

struct conf_item {
    std::shared_ptr<conf_item_spec> spec;
    std::string value;
    conf_item(std::shared_ptr<conf_item_spec> spec, std::string value)
        : spec(std::move(spec)), value(std::move(value)){};
    std::vector<std::string> validate();
};

struct conf_item_mutex {
    std::weak_ptr<conf_class> super_class;
    bool require_one;
    /** if inverse == true the 'mutex' implements: if A then ALSO B, C and D */
    bool inverse;
    std::map<std::string, std::shared_ptr<conf_item_spec>> item_spec_refs;

    void add_item_spec(std::shared_ptr<conf_item_spec>);

    conf_item_mutex(const std::shared_ptr<conf_class> super_class,
                    bool require_one, bool inverse)
        : super_class(super_class), require_one(require_one), inverse(inverse),
          item_spec_refs(){};
};
