#include <fnmatch.h>

int main (int argc , char ** argv) {
  fnmatch( "*" , "/tmp", 0 );
}
