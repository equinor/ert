#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>



int main(void) {
  char key[60];
  char type[60];
  char file[60];
  int active;
  FILE * stream = fopen("file.txt" , "r");
  fscanf(stream , "%s %s" , key , type);
  printf("Posisjon:%d \n",ftell(stream));

  if (fscanf(stream , "%d" , &active) == 1) 
    printf("Har lest actiev = %d \n",active);
  else {
    char c,s1,s2;
    int day,month,year;
    if (fscanf(stream , "%c" , &c) == 1) 
      printf("Har lest char:%c \n",c);
    active = fscanf(stream,"%02d%c%02d%c%4d" , &day , &s1 , &month , &s2 , &year);
    printf("scan_count:%d Data: %02d - %02d - %4d \n",active , day,month,year);
  }
  
  return 0;
}



