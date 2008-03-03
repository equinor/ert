#include <stdio.h>
#include <stdlib.h>

void matlab_write_matrix_ascii(const int nx,const int ny,double *mat,char *filename)
{
 int i,j;
 FILE *stream;
 stream = fopen(filename,"w+");
 if(stream == NULL)
 {
   fprintf(stderr,"%s: failed to open %s for writing - aborting.\n",__func__,filename);
   abort();
 }

 for(i=0; i<nx; i++)
 {
   for(j=0; j<ny ;j++)
   {
     fprintf(stream,"%f ",mat[i*ny+j]);
   }
   fprintf(stream,"\n");
 }

  fclose(stream);
};
