#include <stdlib.h>
#include <unistd.h>

int main(int argc , char ** argv) {
  symlink("/tmp" , "tmp_link");
  exit(1);
}
