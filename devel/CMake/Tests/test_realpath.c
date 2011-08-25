#include <limits.h>
#include <stdlib.h>

int main( int argc , char ** argv) {
  char resolved_path[4096];
  realpath( "/tmp" , resolved_path );
  exit(1);
}
