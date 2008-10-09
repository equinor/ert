#ifndef __CFG_UTIL_H__
#define __CFG_UTIL_H__
#include <stdio.h>
#include <stdbool.h>

void cfg_util_create_token_list(const char *, const char *, const char *, const char *, int *, char ***);
int  cfg_util_sub_size(int, const char **, const char *, const char *);
char * cfg_util_alloc_token_buffer(const char * , const char * , int , const char **);
char * cfg_util_alloc_next_token(char **);
#endif
