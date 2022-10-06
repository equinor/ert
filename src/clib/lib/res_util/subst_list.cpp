#include <filesystem>

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#include <ert/res_util/file_utils.hpp>
#include <ert/util/buffer.hpp>
#include <ert/util/hash.hpp>
#include <ert/util/parser.hpp>
#include <ert/util/util.hpp>
#include <ert/util/vector.hpp>

#include <ert/res_util/subst_func.hpp>
#include <ert/res_util/subst_list.hpp>

namespace fs = std::filesystem;

/*
   This file implements a small support struct for search-replace
   operations, along with wrapped calls to util_string_replace_inplace().

   Substitutions can be carried out on files and string in memory (char *
   with \0 termination); and the operations can be carried out inplace, or
   in a filtering mode where a new file/string is created.


   Usage
   =====
    1. Start with allocating a subst_list instance with subst_list_alloc().

    2. Insert key,value pairs to for search-replace with the functions

        * subst_list_insert_ref(subst_list , key , value);
        * subst_list_insert_owned_ref(subst_list , key , value);
        * subst_list_insert_copy(subst_list , key , value );

       The difference between these functions is who is owning the memory
       pointed to by the value pointer.

    3. Do the actual search-replace operation on a file or memory buffer:

       * subst_list_filter_file()   : Does search-replace on a file.
       * subst_list_update_string() : Does search-replace on a buffer.

    4. Free the subst_list and go home.

   Internally the (key,value) pairs used for substitutions are stored in a
   vector, preserving insert order. If you insert the cascade

     ("A","B")
     ("B","C")
       .....
     ("Y","Z")

   You will eventually end up with a string where all capital letters have
   been transformed to 'Z'.
*/

typedef enum {
    SUBST_DEEP_COPY = 1,
    SUBST_MANAGED_REF = 2,
    SUBST_SHARED_REF = 3
} subst_insert_type; /* Mode used in the subst_list_insert__() function */

#define SUBST_LIST_TYPE_ID 6614320

struct subst_list_struct {
    UTIL_TYPE_ID_DECLARATION;
    /** A parent subst_list instance - can be NULL - no destructor is called
     * for the parent. */
    const subst_list_type *parent;
    /** The string substitutions we should do. */
    vector_type *string_data;
    /** The functions we support. */
    vector_type *func_data;
    /** NOT owned by the subst_list instance - can be NULL */
    const subst_func_pool_type *func_pool;
    hash_type *map;
};

typedef struct {
    /** Pointer to the real subst_func_type - implemented in subst_func.c */
    subst_func_type *func;
    /** The name the function is recognized as - in this substitution context. */
    char *name;
} subst_list_func_type;

/**
   The subst_list type is implemented as a hash of subst_list_node
   instances. This node type is not exported out of this file scope at
   all.
*/
typedef struct {
    /** Wether the memory pointed to by value should bee freed.*/
    bool value_owner;
    char *value;
    char *key;
    /** A doc_string of this substitution - only for documentation - can be NULL. */
    char *doc_string;
} subst_list_string_type;

/** Allocates an empty instance with no values. */
static subst_list_string_type *subst_list_string_alloc(const char *key) {
    subst_list_string_type *node =
        (subst_list_string_type *)util_malloc(sizeof *node);
    node->value_owner = false;
    node->value = NULL;
    node->doc_string = NULL;
    node->key = util_alloc_string_copy(key);
    return node;
}

static void subst_list_string_free_content(subst_list_string_type *node) {
    if (node->value_owner)
        free(node->value);
}

static void subst_list_string_free(subst_list_string_type *node) {
    subst_list_string_free_content(node);
    free(node->doc_string);
    free(node->key);
    free(node);
}

static void subst_list_string_free__(void *node) {
    subst_list_string_free((subst_list_string_type *)node);
}

/**
   input_value can be NULL.
*/
static void subst_list_string_set_value(subst_list_string_type *node,
                                        const char *input_value,
                                        const char *doc_string,
                                        subst_insert_type insert_mode) {
    subst_list_string_free_content(node);
    {
        char *value;
        if (insert_mode == SUBST_DEEP_COPY)
            value = util_alloc_string_copy(input_value);
        else
            value = (char *)input_value;

        if (insert_mode == SUBST_SHARED_REF)
            node->value_owner = false;
        else
            node->value_owner = true;

        node->value = value;
    }

    if (doc_string != NULL)
        node->doc_string =
            util_realloc_string_copy(node->doc_string, doc_string);
}

/**
   When arriving at this function the main subst scope should already have verified that
   the requested function is available in the function pool.
*/
static subst_list_func_type *subst_list_func_alloc(const char *func_name,
                                                   subst_func_type *func) {
    subst_list_func_type *subst_func =
        (subst_list_func_type *)util_malloc(sizeof *subst_func);
    subst_func->name = util_alloc_string_copy(func_name);
    subst_func->func = func;
    return subst_func;
}

static void subst_list_func_free(subst_list_func_type *subst_func) {
    free(subst_func->name);
    free(subst_func);
}

static void subst_list_func_free__(void *node) {
    subst_list_func_free((subst_list_func_type *)node);
}

static char *subst_list_func_eval(const subst_list_func_type *subst_func,
                                  const stringlist_type *arglist) {
    return subst_func_eval(subst_func->func, arglist);
}

/**
   Find the node corresponding to 'key' -  returning NULL if it is not found.
*/
static subst_list_string_type *
subst_list_get_string_node(const subst_list_type *subst_list, const char *key) {
    subst_list_string_type *node = NULL;
    int index = 0;

    /* Linear search ... */ /*Should use map*/
    while ((index < vector_get_size(subst_list->string_data)) &&
           (node == NULL)) {
        subst_list_string_type *inode = (subst_list_string_type *)vector_iget(
            subst_list->string_data, index);

        if (strcmp(inode->key, key) == 0) /* Found it */
            node = inode;
        else
            index++;
    }

    return node;
}

static subst_list_string_type *
subst_list_insert_new_node(subst_list_type *subst_list, const char *key,
                           bool append) {
    subst_list_string_type *new_node = subst_list_string_alloc(key);

    if (append)
        vector_append_owned_ref(subst_list->string_data, new_node,
                                subst_list_string_free__);
    else
        vector_insert_owned_ref(subst_list->string_data, 0, new_node,
                                subst_list_string_free__);

    hash_insert_ref(subst_list->map, key, new_node);
    return new_node;
}

UTIL_IS_INSTANCE_FUNCTION(subst_list, SUBST_LIST_TYPE_ID)

/**
   Observe that this function sets both the subst parent, and the pool
   of available functions. If this is call is repeated it is possible
   to create a weird config with dangling function pointers - that is
   a somewhat contrived and pathological use case, and not checked
   for.
*/
void subst_list_set_parent(subst_list_type *subst_list,
                           const subst_list_type *parent) {
    subst_list->parent = parent;
    if (parent != NULL)
        subst_list->func_pool = subst_list->parent->func_pool;
}

bool subst_list_has_key(const subst_list_type *subst_list, const char *key) {
    return hash_has_key(subst_list->map, key);
}

/**
   The input argument is currently only (void *), runtime it will be
   checked whether it is of type subst_list_type, in which case it is
   interpreted as a parent instance, if it is of type
   subst_func_pool_type it is interpreted as a func_pool instance,
   otherwise the function will fail hard.

   If the the input argument is a subst_list parent, the func_pool of
   the parent is used also for the newly allocated subst_list
   instance.
*/
subst_list_type *subst_list_alloc(const void *input_arg) {
    subst_list_type *subst_list =
        (subst_list_type *)util_malloc(sizeof *subst_list);
    UTIL_TYPE_ID_INIT(subst_list, SUBST_LIST_TYPE_ID);
    subst_list->parent = NULL;
    subst_list->func_pool = NULL;
    subst_list->map = hash_alloc();
    subst_list->string_data = vector_alloc_new();
    subst_list->func_data = vector_alloc_new();

    if (input_arg != NULL) {
        if (subst_list_is_instance(input_arg))
            subst_list_set_parent(subst_list,
                                  (const subst_list_type *)input_arg);
        else if (subst_func_pool_is_instance(input_arg))
            subst_list->func_pool = (const subst_func_pool_type *)input_arg;
        else
            util_abort(
                "%s: run_time cast failed - invalid type on input argument.\n",
                __func__);
    }

    return subst_list;
}

/**
   The semantics of the doc_string string is as follows:

    1. If doc_string is different from NULL the doc_string is stored in the node.

    2. If a NULL value follows a non-NULL value (i.e. the substitution
       is updated) the doc_string is not changed.

   The idea is that the doc_string must just be included the first
   time a (key,value) pair is added. On subsequent updates of the
   value, the doc_string string can be NULL.
*/
static void subst_list_insert__(subst_list_type *subst_list, const char *key,
                                const char *value, const char *doc_string,
                                bool append, subst_insert_type insert_mode) {
    subst_list_string_type *node = subst_list_get_string_node(subst_list, key);

    if (node == NULL) /* Did not have the node. */
        node = subst_list_insert_new_node(subst_list, key, append);
    subst_list_string_set_value(node, value, doc_string, insert_mode);
}

/*
   There are three different functions for inserting a key-value pair
   in the subst_list instance. The difference between the three is in
   which scope takes/has ownership of 'value'. The alternatives are:

    subst_list_insert_ref: In this case the calling scope has full
       ownership of value, and is consquently responsible for freeing
       it, and ensuring that it stays a valid pointer for the subst_list
       instance. Probably the most natural function to use when used
       with static storage, i.e. typically string literals.

    subst_list_insert_owned_ref: In this case the subst_list takes
       ownership of the value reference, in the sense that it will
       free it when it is done.

    subst_list_insert_copy: In this case the subst_list takes a copy
       of value and inserts it. Meaning that the substs_list instance
       takes repsonibility of freeing, _AND_ the calling scope is free
       to do whatever it wants with the value pointer.

*/

void subst_list_append_owned_ref(subst_list_type *subst_list, const char *key,
                                 const char *value, const char *doc_string) {
    subst_list_insert__(subst_list, key, value, doc_string, true,
                        SUBST_MANAGED_REF);
}

void subst_list_append_copy(subst_list_type *subst_list, const char *key,
                            const char *value, const char *doc_string) {
    subst_list_insert__(subst_list, key, value, doc_string, true,
                        SUBST_DEEP_COPY);
}

void subst_list_prepend_ref(subst_list_type *subst_list, const char *key,
                            const char *value, const char *doc_string) {
    subst_list_insert__(subst_list, key, value, doc_string, false,
                        SUBST_SHARED_REF);
}

void subst_list_prepend_owned_ref(subst_list_type *subst_list, const char *key,
                                  const char *value, const char *doc_string) {
    subst_list_insert__(subst_list, key, value, doc_string, false,
                        SUBST_MANAGED_REF);
}

void subst_list_prepend_copy(subst_list_type *subst_list, const char *key,
                             const char *value, const char *doc_string) {
    subst_list_insert__(subst_list, key, value, doc_string, false,
                        SUBST_DEEP_COPY);
}

/**
   This function will install the function @func_name from the current
   subst_func_pool, it will be made available to this subst_list
   instance with the function name @local_func_name. If @func_name is
   not available, the function will fail hard.
*/
void subst_list_insert_func(subst_list_type *subst_list, const char *func_name,
                            const char *local_func_name) {

    if (subst_list->func_pool != NULL &&
        subst_func_pool_has_func(subst_list->func_pool, func_name)) {
        subst_list_func_type *subst_func = subst_list_func_alloc(
            local_func_name,
            subst_func_pool_get_func(subst_list->func_pool, func_name));
        vector_append_owned_ref(subst_list->func_data, subst_func,
                                subst_list_func_free__);
    } else
        util_abort("%s: function:%s not available \n", __func__, func_name);
}

void subst_list_clear(subst_list_type *subst_list) {
    vector_clear(subst_list->string_data);
}

void subst_list_free(subst_list_type *subst_list) {
    vector_free(subst_list->string_data);
    vector_free(subst_list->func_data);
    hash_free(subst_list->map);
    free(subst_list);
}

/*
  Below comes many different functions for doing the actual updating
  the functions differ in the form of the input and output. At the
  lowest level, is the function

    subst_list_uppdate_buffer()

  which will update a buffer instance. This function again will call
  two separate functions for pure string substitutions and for
  function evaluation.

  The update replace functions will first apply all the string
  substitutions for this particular instance, and afterwards calling
  all the string substititions of the parent (recursively); the same
  applies to the function replacements.
*/

/**
   Updates the buffer inplace with all the string substitutions in the
   subst_list. This is the lowest level function, which does *NOT*
   consider the parent pointer.
*/
static bool subst_list_replace_strings__(const subst_list_type *subst_list,
                                         buffer_type *buffer) {
    int index;
    bool global_match = false;
    for (index = 0; index < vector_get_size(subst_list->string_data); index++) {
        const subst_list_string_type *node =
            (const subst_list_string_type *)vector_iget_const(
                subst_list->string_data, index);
        if (node->value != NULL) {
            bool match;
            buffer_rewind(buffer);
            do {
                match = buffer_search_replace(buffer, node->key, node->value);
                if (match)
                    global_match = true;
            } while (match);
        }
    }
    return global_match;
}

/**
   Updates the buffer inplace by evaluationg all the string functions
   in the subst_list. Last performing all the replacements in the
   parent.

   The rules for function evaluation are as follows:

    1. Every function MUST have a '()', IMMEDIATELY following the
       function name, this also applies to functions which do not have
       any arguments.

    2. The function parser is quite primitive, and can (at least
       currently) not handle nested functions, i.e.

          __SUM__( __SUM__(1 ,2 ) , 3)

       will fail.

    3. If the function evaluation fails; typicall because of wrong
       type/number of arguments the buffer will not updated.


    4. The functions will return a freshly allocated (char *) pointer,
       or NULL if the evaluation fails, and the input value is a list
       of strings extracted from the parsing context - i.e. the
       function is in no way limited to numeric functions like sum()
       and exp().

*/
static bool subst_list_eval_funcs____(const subst_list_type *subst_list,
                                      const basic_parser_type *parser,
                                      buffer_type *buffer) {
    bool global_match = false;
    int index;
    for (index = 0; index < vector_get_size(subst_list->func_data); index++) {
        const subst_list_func_type *subst_func =
            (const subst_list_func_type *)vector_iget_const(
                subst_list->func_data, index);
        const char *func_name = subst_func->name;

        bool match;
        buffer_rewind(buffer);
        do {
            size_t match_pos;
            match = buffer_strstr(buffer, func_name);
            match_pos = buffer_get_offset(buffer);

            if (match) {
                bool update = false;
                char *arg_start = (char *)buffer_get_data(buffer);
                arg_start += buffer_get_offset(buffer) + strlen(func_name);

                if (arg_start[0] == '(') {
                    // We require that an opening paren follows immediately
                    // behind the function name.
                    char *arg_end = strchr(arg_start, ')');
                    if (arg_end != NULL) {
                        /* OK - we found an enclosing () pair. */
                        char *arg_content = util_alloc_substring_copy(
                            arg_start, 1, arg_end - arg_start - 1);
                        stringlist_type *arg_list =
                            basic_parser_tokenize_buffer(parser, arg_content,
                                                         true);
                        char *func_eval =
                            subst_list_func_eval(subst_func, arg_list);
                        int old_len =
                            strlen(func_name) + strlen(arg_content) + 2;

                        if (func_eval != NULL) {
                            buffer_memshift(buffer, match_pos + old_len,
                                            strlen(func_eval) - old_len);
                            buffer_fwrite(buffer, func_eval, strlen(func_eval),
                                          sizeof *func_eval);
                            free(func_eval);
                            update = true;
                            global_match = true;
                        }

                        free(arg_content);
                        stringlist_free(arg_list);
                    }
                }
                if (!update)
                    buffer_fseek(buffer, match_pos + strlen(func_name),
                                 SEEK_SET);
            }
        } while (match);
    }
    if (subst_list->parent != NULL)
        global_match =
            (subst_list_eval_funcs____(subst_list->parent, parser, buffer) ||
             global_match);
    return global_match;
}

static bool subst_list_eval_funcs__(const subst_list_type *subst_list,
                                    buffer_type *buffer) {
    basic_parser_type *parser =
        basic_parser_alloc(",", "\"\'", NULL, " \t", NULL, NULL);
    bool match = subst_list_eval_funcs____(subst_list, parser, buffer);

    basic_parser_free(parser);
    return match;
}

/**
   Should we evaluate the parent first (i.e. top down), or this
   instance first and then subsequently the parent (i.e. bottom
   up). The problem is with inherited defintions:

     Inherited defintions
     --------------------

     In this situation we have defined a (key,value) substitution,
     where the value depends on a value following afterwards:

       ("<PATH>" , "/tmp/run/<CASE>")
       ("<CASE>" , "Test4")

     I.e. first <PATH> is replaced with "/tmp/run/<CASE>" and then
     subsequently "<CASE>" is replaced with "Test4". A typical use
     case here is that the common definition of "<PATH>" is in the
     parent, and consequently parent should run first (i.e. top
     down).

     However, in other cases the order of defintion might very well be
     opposite, i.e. with "<CASE>" first and then things will blow up:

       1. <CASE>: Not found
       2. <PATH> -> /tmp/run/<CASE>

     and, the final <CASE> will not be resolved. I.e. there is no
     obvious 'right' way to do it.



     Overriding defaults
     -------------------

     The parent has defined:

        ("<PATH>" , "/tmp/run/default")

     But for a particular instance we would like to overwrite <PATH>
     with another definition:

        ("<PATH>" , "/tmp/run/special_case")

     This will require evaluating the bottom first, i.e. a bottom up
     approach.


   Currently the implementation is purely top down, the latter case
   above is not supported. The actual implementation here is in terms
   of recursion, the low level function doing the stuff is
   subst_list_replace_strings__() which is not recursive.
*/
static bool subst_list_replace_strings(const subst_list_type *subst_list,
                                       buffer_type *buffer) {
    bool match = false;
    if (subst_list->parent != NULL)
        match = subst_list_replace_strings(subst_list->parent, buffer);

    /* The actual string replace */
    match = (subst_list_replace_strings__(subst_list, buffer) || match);
    return match;
}

/**
  This function updates a buffer instance inplace with all the
  substitutions in the subst_list.

  This is the common low-level function employed by all the the
  subst_update_xxx() functions. Observe that it is a hard assumption
  that the buffer has a \0 terminated string.
*/
bool subst_list_update_buffer(const subst_list_type *subst_list,
                              buffer_type *buffer) {
    bool match1 = subst_list_replace_strings(subst_list, buffer);
    bool match2 = subst_list_eval_funcs__(subst_list, buffer);
    // Funny construction to ensure to avoid fault short circuit:
    return (match1 || match2);
}

/**
   This function reads the content of a file, and writes a new file
   where all substitutions in subst_list have been performed. Observe
   that target_file and src_file *CAN* point to the same file, in
   which case this will amount to an inplace update. In that case a
   backup file is written, and held, during the execution of the
   function.

   Observe that @target_file can contain a path component, that
   component will be created if it does not exist.
*/
bool subst_list_filter_file(const subst_list_type *subst_list,
                            const char *src_file, const char *target_file) {
    bool match;
    char *backup_file = NULL;
    buffer_type *buffer = buffer_fread_alloc(src_file);
    // Ensure that the buffer is a \0 terminated string:
    buffer_fseek(buffer, 0, SEEK_END);
    buffer_fwrite_char(buffer, '\0');

    if (util_same_file(src_file, target_file)) {
        char *backup_prefix = util_alloc_sprintf("%s-%s", src_file, __func__);
        backup_file = util_alloc_tmp_file("/tmp", backup_prefix, false);
        free(backup_prefix);
    }

    /* Writing backup file */
    if (backup_file != NULL) {
        FILE *stream = util_fopen(backup_file, "w");
        buffer_stream_fwrite_n(buffer, 0, -1,
                               stream); /* -1: Do not write the trailing \0. */
        fclose(stream);
    }

    /* Doing the actual update */
    match = subst_list_update_buffer(subst_list, buffer);

    /* Writing updated file */
    {
        auto stream = mkdir_fopen(fs::path(target_file), "w");

        buffer_stream_fwrite_n(buffer, 0, -1,
                               stream); /* -1: Do not write the trailing \0. */
        fclose(stream);
    }

    /* OK - all went hunka dory - unlink the backup file and leave the building. */
    if (backup_file != NULL) {
        remove(backup_file);
        free(backup_file);
    }
    buffer_free(buffer);
    return match;
}

/**
   This function does search-replace on string instance inplace.
*/
bool subst_list_update_string(const subst_list_type *subst_list,
                              char **string) {
    buffer_type *buffer =
        buffer_alloc_private_wrapper(*string, strlen(*string) + 1);
    bool match = subst_list_update_buffer(subst_list, buffer);
    *string = (char *)buffer_get_data(buffer);
    buffer_free_container(buffer);

    return match;
}

/**
   This function allocates a new string where the search-replace
   operation has been performed.
*/
char *subst_list_alloc_filtered_string(const subst_list_type *subst_list,
                                       const char *string) {
    char *filtered_string = util_alloc_string_copy(string);
    if (subst_list)
        subst_list_update_string(subst_list, &filtered_string);
    return filtered_string;
}

/**
   This allocates a new subst_list instance, the copy process is deep,
   in the sense that all srings inserted in the new subst_list
   instance have their own storage, irrespective of the ownership in
   the original subst_list instance.
*/
subst_list_type *subst_list_alloc_deep_copy(const subst_list_type *src) {
    subst_list_type *copy;
    if (src->parent != NULL)
        copy = subst_list_alloc(src->parent);
    else
        copy = subst_list_alloc(src->func_pool);

    {
        int index;
        for (index = 0; index < vector_get_size(src->string_data); index++) {
            const subst_list_string_type *node =
                (const subst_list_string_type *)vector_iget_const(
                    src->string_data, index);
            subst_list_insert__(copy, node->key, node->value, node->doc_string,
                                true, SUBST_DEEP_COPY);
        }

        for (index = 0; index < vector_get_size(src->func_data); index++) {
            const subst_list_func_type *src_node =
                (const subst_list_func_type *)vector_iget_const(src->func_data,
                                                                index);
            subst_list_func_type *copy_node =
                subst_list_func_alloc(src_node->name, src_node->func);
            vector_append_owned_ref(copy->func_data, copy_node,
                                    subst_list_func_free__);
        }
    }
    return copy;
}

int subst_list_get_size(const subst_list_type *subst_list) {
    return vector_get_size(subst_list->string_data);
}

const char *subst_list_iget_key(const subst_list_type *subst_list, int index) {
    if (index < vector_get_size(subst_list->string_data)) {
        const subst_list_string_type *node =
            (const subst_list_string_type *)vector_iget_const(
                subst_list->string_data, index);
        return node->key;
    } else {
        util_abort("%s: index:%d to large \n", __func__, index);
        return NULL;
    }
}

const char *subst_list_iget_value(const subst_list_type *subst_list,
                                  int index) {
    if (index < vector_get_size(subst_list->string_data)) {
        const subst_list_string_type *node =
            (const subst_list_string_type *)vector_iget_const(
                subst_list->string_data, index);
        return node->value;
    } else {
        util_abort("%s: index:%d to large \n", __func__, index);
        return NULL;
    }
}

const char *subst_list_get_value(const subst_list_type *subst_list,
                                 const char *key) {
    const subst_list_string_type *node =
        (const subst_list_string_type *)hash_get(subst_list->map, key);
    return node->value;
}

const char *subst_list_get_doc_string(const subst_list_type *subst_list,
                                      const char *key) {
    const subst_list_string_type *node =
        (const subst_list_string_type *)hash_get(subst_list->map, key);
    return node->doc_string;
}

void subst_list_fprintf(const subst_list_type *subst_list, FILE *stream) {
    int index;
    for (index = 0; index < vector_get_size(subst_list->string_data); index++) {
        const subst_list_string_type *node =
            (const subst_list_string_type *)vector_iget_const(
                subst_list->string_data, index);
        fprintf(stream, "%s = %s\n", node->key, node->value);
    }
}

/**
   This function splits a string on the given character, taking into account
   that it may contain strings delimited by ' or ". It returns the length of the
   first part of the splitted string. For instance, splitting on a ','
   character, if the input is:

   foo "blah, foo" x 'y', and more "stuff"

   it will find the comma after 'y', and hence return a value of 21.

   The function returns a negative value if it contains a string, started with '
   or ", which is not terminated. It returns the length of the string if the
   split character is not found.
*/
static int find_substring(const char *arg_string, const char *split_char) {
    char pattern[4] = {'"', '\'', '\0',
                       '\0'}; // we accept both ' and " to delimite strings.
    strcat(pattern, split_char);
    int len = strcspn(arg_string, pattern);

    // If string delimiter is found, we need to find the corresponding end
    // delimiter. If we do not find it, that is an error. If we do find it, we
    // need to continue searching for the split character or the start of
    // another string.
    while (strlen(arg_string) > len &&
           (arg_string[len] == '"' || arg_string[len] == '\'')) {
        const char delimiter = arg_string[len];

        // Add the delimiter to the length found so far.
        len++;

        // The string must be long enough to accomdate a corresponding delimiter.
        if (strlen(arg_string) <= len)
            return -1;

        // Find the corresponding delimiter, start searching the string right
        // after the delimiter we just found, at an offset of len.
        const char *end = strchr(arg_string + len + 1, delimiter);

        // No corresponding end delimiter is an error.
        if (end == NULL)
            return -1;

        // Update the lenght of the substring found so far.
        len = end - arg_string + 1;

        // We found the second string delimiter, but not the character we are
        // using for splitting. Therefore, repeat the original search starting
        // right after the second delimiter. We keep doing this in a loop until
        // we find the split character, or until the string is exhausted.
        if (strlen(end + 1) > 0)
            len += strcspn(end + 1, pattern);
    }

    return len;
}

/**
   Trim spaces left and right from a string. Do not reallocate the string, just
   move the start pointer of the string, and terminate the string appropiately.
*/
static char *trim_string(char *str) {
    // Move the start of the string to skip space.
    while (isspace(*str))
        str++;
    // Shorten the string to remove trailing space.
    int len = strlen(str);
    while (len > 0 && isspace(str[len - 1]))
        len--;
    str[len] = '\0';
    return str;
}

void subst_list_add_from_string(subst_list_type *subst_list,
                                const char *arg_string_orig, bool append) {
    if (!arg_string_orig)
        return;

    // Copy the string, since we will modify it while working on it, and trim it.
    char *arg_string_copy = util_alloc_string_copy(arg_string_orig);
    char *arg_string = trim_string(arg_string_copy);
    char *tmp = NULL;

    while (strlen(arg_string)) {
        // Find the next argument/value pair, by splitting on a ','.
        int arg_len = find_substring(arg_string, ",");
        if (arg_len < 0)
            util_abort("%s: missing string delimiter in argument: %s\n",
                       __func__, arg_string_orig);

        // Extract the argument/value pair, and parse it.
        tmp = util_alloc_substring_copy(arg_string, 0, arg_len);

        // Split on '=' to find the argument name (key) and value.
        int key_len = find_substring(tmp, "=");

        if (key_len < 0) // There is a ' or " string that is not closed.
            util_abort("%s: missing string delimiter in argument: %s\n",
                       __func__, arg_string_orig);
        if (key_len == strlen(tmp)) // There is no '=".
            util_abort("%s: missing '=' in argument: %s\n", __func__,
                       arg_string_orig);

        // Split the string into trimmed key and value strings.
        tmp[key_len] = '\0';
        char *key = trim_string(tmp);
        char *value = trim_string(tmp + key_len + 1);

        // Check that the key and value strings are not empty.
        if (strlen(key) == 0)
            util_abort("%s: missing key in argument list: %s\n", __func__,
                       arg_string_orig);
        if (strlen(value) == 0)
            util_abort("%s: missing value in argument list: %s\n", __func__,
                       arg_string_orig);

        // Check that the key does not contain string delimiters.
        if (strchr(key, '\'') || strchr(key, '"'))
            util_abort("%s: key cannot be a string: %s\n", __func__,
                       arg_string_orig);

        // Add to the list of parsed arguments.
        if (append)
            subst_list_append_copy(subst_list, key, value, NULL);
        else
            subst_list_prepend_copy(subst_list, key, value, NULL);

        free(tmp);

        // Skip to the part of the string that was not parsed yet.
        arg_string += arg_len;

        // Skip whitespace and at most one comma.
        arg_string = trim_string(arg_string);
        if (*arg_string == ',') {
            arg_string = trim_string(arg_string + 1);
            if (strlen(arg_string) == 0) // trailing comma.
                util_abort("%s: trailing comma in argument list: %s\n",
                           __func__, arg_string_orig);
        }
    }

    free(arg_string_copy);
}
