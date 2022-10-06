#ifndef ERT_PATH_FMT_H
#define ERT_PATH_FMT_H

#include <stdarg.h>
#include <stdbool.h>

#include <ert/util/node_ctype.hpp>

typedef struct path_fmt_struct path_fmt_type;

extern "C" path_fmt_type *path_fmt_alloc_directory_fmt(const char *);
path_fmt_type *path_fmt_alloc_path_fmt(const char *);
char *path_fmt_alloc_path(const path_fmt_type *, bool, ...);
char *path_fmt_alloc_file(const path_fmt_type *, bool, ...);
extern "C" void path_fmt_free(path_fmt_type *);
void path_fmt_free__(void *arg);
extern "C" const char *path_fmt_get_fmt(const path_fmt_type *);
void path_fmt_reset_fmt(path_fmt_type *, const char *);
path_fmt_type *path_fmt_realloc_path_fmt(path_fmt_type *path_fmt,
                                         const char *fmt);

#endif
