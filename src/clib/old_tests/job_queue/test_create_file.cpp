#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    char *filename = argv[1];
    int value = atoi(argv[2]);
    FILE *stream = fopen(filename, "w");
    fprintf(stream, "%d\n", value);
    fclose(stream);
}
