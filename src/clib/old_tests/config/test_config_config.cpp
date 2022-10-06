#include <stdlib.h>

#include <ert/config/config_parser.hpp>

int main(int argc, char **argv) {
    config_parser_type *config = config_alloc();
    config_add_schema_item(config, "KEYWORD", false);
    config_free(config);
    exit(0);
}
