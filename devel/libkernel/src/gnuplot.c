#include <stdio.h>
#include <stdlib.h>

#include <util.h>



void gnuplot_write_gridded(const int nx,const int ny,const double *gridx, const double *gridy,const double *z, char *filename)
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
      fprintf(stream,"%f %f %f\n",gridx[i],gridy[j],z[i*ny+j]);
    }
    fprintf(stream,"\n");
  }

  fclose(stream);
};



void gnuplot_write_ungridded(const int nx,const int ny, const double *z, char *filename)
{
  double *gridx;
  double *gridy;

  int i;

  gridx = util_malloc(nx * sizeof *gridx,__func__);
  gridy = util_malloc(ny * sizeof *gridy,__func__);

  for(i=0; i<nx; i++)
  {
    gridx[i] = (double) i;
  }
  for(i=0; i<ny; i++)
  {
    gridy[i] = (double) i;
  }

  gnuplot_write_gridded(nx,ny,gridx,gridy,z,filename);

  free(gridx);
  free(gridy);
};
