#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv) {
    int max_count = 100;
    int count = 0;
    while (true) {
        sleep(1);
        count++;
        printf("%d/%d \n", count, max_count);
        if (count == max_count)
            break;
    }
    exit(0);
}
