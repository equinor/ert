#include <time.h>
#include <sys/types.h>

int main(int argc, char ** argv) {
  struct tm ts;
  time_t t;

  localtime_r( &t , &ts );
}

