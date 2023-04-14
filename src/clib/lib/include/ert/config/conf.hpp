#ifndef ERT_CONF_H
#define ERT_CONF_H

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
#include <set>
#include <string>
#include <vector>

using conf_class_type = struct conf_class_struct;
using conf_instance_type = struct conf_instance_struct;
using conf_item_spec_type = struct conf_item_spec_struct;
using conf_item_type = struct conf_item_struct;
using conf_item_mutex_type = struct conf_item_mutex_struct;

struct conf_class_struct {
    /** Can be NULL */
    std::shared_ptr<conf_class_type> super_class;
    char *class_name;
    /** Can be NULL if not given */
    char *help;
    bool require_instance;
    bool singleton;

    std::map<std::string, std::shared_ptr<conf_class_type>> sub_classes;
    std::map<std::string, std::shared_ptr<conf_item_spec_type>> item_specs;
    std::vector<std::shared_ptr<conf_item_mutex_type>> item_mutexes;
};

struct conf_instance_struct {
    std::shared_ptr<conf_class_type> conf_class;
    char *name;

    std::map<std::string, std::shared_ptr<conf_instance_type>> sub_instances;
    std::map<std::string, std::shared_ptr<conf_item_type>> items;
};

struct conf_item_spec_struct {
    /** NULL if not inserted into a class */
    std::shared_ptr<conf_class_type> super_class;
    char *name;
    /** Require the item to take a valid value */
    bool required_set;
    /** Can be NULL if not given */
    char *default_value;
    /** Data type. See conf_data */
    dt_enum dt;
    std::set<std::string> *restriction;
    /** Can be NULL if not given */
    char *help;
};

struct conf_item_struct {
    std::shared_ptr<conf_item_spec_type> conf_item_spec;
    char *value;
};

struct conf_item_mutex_struct {
    std::shared_ptr<conf_class_type> super_class;
    bool require_one;
    /** if inverse == true the 'mutex' implements: if A then ALSO B, C and D */
    bool inverse;
    std::map<std::string, std::shared_ptr<conf_item_spec_type>> item_spec_refs;
};

/** D E F A U L T   A L L O C / F R E E    F U N C T I O N S */

std::shared_ptr<conf_class_type> make_conf_class(const char *class_name,
                                                 bool require_instance,
                                                 bool singleton,
                                                 const char *help);
std::shared_ptr<conf_item_spec_type> make_conf_item_spec(const char *name,
                                                         bool required_set,
                                                         dt_enum dt,
                                                         const char *help);
/** M A N I P U L A T O R S ,   I N S E R T I O N */

void conf_class_insert_sub_class(
    std::shared_ptr<conf_class_type> conf_class,
    std::shared_ptr<conf_class_type> sub_conf_class);

void conf_class_insert_item_spec(
    std::shared_ptr<conf_class_type> conf_class,
    std::shared_ptr<conf_item_spec_type> item_spec);

void conf_instance_insert_sub_instance(
    std::shared_ptr<conf_instance_type> conf_instance,
    std::shared_ptr<conf_instance_type> sub_conf_instance);

void conf_instance_insert_item(conf_instance_type *conf_instance,
                               const char *item_name, const char *value);

std::shared_ptr<conf_item_mutex_type>
conf_class_new_item_mutex(std::shared_ptr<conf_class_type> conf_class,
                          bool require_one, bool inverse);

void conf_item_mutex_add_item_spec(
    std::shared_ptr<conf_item_mutex_type> conf_item_mutex,
    std::shared_ptr<conf_item_spec_type> conf_item_spec);

/** M A N I P U L A T O R S ,   C L A S S   A N D   I T E M   S P E C I F I C A T I O N */

void conf_class_set_help(std::shared_ptr<conf_class_type> conf_class,
                         const char *help);

void conf_item_spec_add_restriction(
    std::shared_ptr<conf_item_spec_type> conf_item_spec,
    const char *restriction);

void conf_item_spec_set_default_value(
    std::shared_ptr<conf_item_spec_type> conf_item_spec,
    const char *default_value);

/** A C C E S S O R S */

bool conf_class_has_item_spec(std::shared_ptr<conf_class_type> conf_class,
                              const char *item_name);

bool conf_class_has_sub_class(std::shared_ptr<conf_class_type> conf_class,
                              const char *sub_class_name);

std::shared_ptr<conf_class_type>
conf_class_get_sub_class_ref(std::shared_ptr<conf_class_type> conf_class,
                             const char *sub_class_name);

const char *
conf_instance_get_name_ref(std::shared_ptr<conf_instance_type> conf_instance);

bool conf_instance_is_of_class(
    std::shared_ptr<conf_instance_type> conf_instance, const char *class_name);

bool conf_instance_has_item(std::shared_ptr<conf_instance_type> conf_instance,
                            std::string item_name);

std::shared_ptr<conf_instance_type> conf_instance_get_sub_instance_ref(
    std::shared_ptr<conf_instance_type> conf_instance,
    const char *sub_instance_name);

std::vector<std::string>
conf_instance_alloc_list_of_sub_instances_of_class_by_name(
    std::shared_ptr<conf_instance_type> conf_instance,
    const char *sub_class_name);

const char *conf_instance_get_class_name_ref(
    std::shared_ptr<conf_instance_type> conf_instance);

const char *conf_instance_get_item_value_ref(
    std::shared_ptr<conf_instance_type> conf_instance, std::string item_name);

/** If the dt supports it, these functions will parse the item
    value to the requested types.

    NOTE:
    If the dt does not support it, or the conf_instance
    does not have the item, the functions will abort your program.
*/
int conf_instance_get_item_value_int(
    std::shared_ptr<conf_instance_type> conf_instance, std::string item_name);
double conf_instance_get_item_value_double(
    std::shared_ptr<conf_instance_type> conf_instance, std::string item_name);

time_t conf_instance_get_item_value_time_t(
    std::shared_ptr<conf_instance_type> conf_instance, std::string item_name);

char *
conf_instance_get_path_error(std::shared_ptr<conf_instance_type> conf_instance);

/** V A L I D A T O R S */

bool conf_instance_validate(std::shared_ptr<conf_instance_type> conf_instance);

/** A L L O C   F R O M   F I L E */

std::shared_ptr<conf_instance_type>
conf_instance_alloc_from_file(std::shared_ptr<conf_class_type> conf_class,
                              const char *name, const char *file_name);

#endif
