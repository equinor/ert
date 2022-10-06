#ifndef ERT_CONF_UTIL_H
#define ERT_CONF_UTIL_H

char *conf_util_fscanf_alloc_token_buffer(const char *file_name);

char *conf_util_alloc_next_token(char **buffer_position);

#endif
