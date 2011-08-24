#include <sys/types.h>
#include <unistd.h>

int main(int argc , char ** argv) {
  pid_t child_pid = fork( );
  BUG
  exit(1);
}
