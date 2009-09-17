#include <util.h>
#include <enkf_main.h>
#include <set.h>
#include <unistd.h>


int main (int argc , char ** argv) {
  char hostname[256];
  const char * executable = argv[1];
  
  gethostname( hostname , 255 );
  printf("%s : " , hostname);
  {
    set_type * user_set = set_alloc_empty();
    enkf_main_list_users( user_set , executable );
    
    if (set_get_size( user_set ) > 0) 
      set_fprintf(user_set , " " , stdout );
    else
      printf("No users.");
    printf("\n");
    
    set_free( user_set );
  }
}




