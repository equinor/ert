#include <glob.h>
#include <util.h>

int main (int argc , char ** argv) {
  glob_t * pglob = util_malloc( sizeof * pglob , __func__);
  int glob_flags = 0;
  glob( "/tmp/*" , glob_flags , NULL , pglob);
  globfree( pglob );
}
