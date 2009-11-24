#include <stdbool.h>
#include <util.h>
#include <sched_file.h>
#include <history.h>
#include <sched_kw_wconinje.h>




int main(int argc, char **argv)
{
  printf("File: %s contains:%d report steps \n",argv[1] , sched_file_step_count( argv[1] ));
}
