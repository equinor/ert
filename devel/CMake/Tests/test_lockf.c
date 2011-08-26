#include <unistd.h>


int main(int argc, char ** argv) {
  int fd = open( "/tmp/lock" , O_WRONLY + O_CREAT , mode); 
  lockf( fd , F_TLOCK , 0);
}

