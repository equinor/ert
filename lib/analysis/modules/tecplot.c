void static teccost(const matrix_type * W,
             const matrix_type * D,
             const char * fname,
             const int ens_size,
             const int izone)
{
   FILE *fp;
   float costJ[ens_size];

   if (izone == 1){
      fp = fopen(fname, "w");
      fprintf(fp, "TITLE = \"%s\"\n",fname);
      fprintf(fp, "VARIABLES = \"i\" \"Average J\" ");
      for (int i = 0; i < ens_size; i++)
         fprintf(fp, "\"%d\" ",i);
      fprintf(fp,"\n");
   } else {
      fp = fopen(fname, "a");
   }

   float costf=0.0;
   for (int i = 0; i < ens_size; i++){
      costJ[i]=matrix_column_column_dot_product(W,i,W,i)+matrix_column_column_dot_product(D,i,D,i);
      costf += costJ[i];
   }
   costf= costf/ens_size;

   fprintf(fp,"%2d %10.3f ", izone-1, costf) ;
   for (int i = 0; i < ens_size; i++)
      fprintf(fp,"%10.3f ", costJ[i] ) ;
   fprintf(fp,"\n") ;
   fclose(fp);
}

void static teclog(const matrix_type * W,
            const matrix_type * D,
            const matrix_type * DW,
            const char * fname,
            const int ens_size,
            const int itnr,
            const double rcond,
            const int nrsing,
            const int nrobs)
{

   double diff1W = 0.0;
   double diff2W = 0.0;
   double diffW;
   double costfp,costfd,costf;
   FILE *fp;

   if (itnr == 1){
      fp = fopen(fname, "w");
      fprintf(fp, "TITLE = \"%s\"\n",fname);
      fprintf(fp, "VARIABLES = \"it\" \"Diff1W\" \"Diff2W\" \"J-prior\" \"J-data\" \"J\" \"rcond\" \"Sing\" \"nrobs\"\n");
   } else {
      fp = fopen(fname, "a");
   }

   for (int i = 0; i < ens_size; i++){
       diffW=matrix_column_column_dot_product(DW,i,DW,i)/ens_size;
       diff2W+=diffW;
       if (diffW > diff1W) diff1W=diffW ;
   }
   diff2W=diff2W/ens_size;

   costfp=0.0;
   costfd=0.0;
   for (int i = 0; i < ens_size; i++){
       costfp += matrix_column_column_dot_product(W,i,W,i);
       costfd += matrix_column_column_dot_product(D,i,D,i);
   }
   costfp=costfp/ens_size;
   costfd=costfd/ens_size;
   costf= costfp + costfd;
   fprintf(fp," %d %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %d %d\n", itnr-1, diff1W, diff2W, costfp, costfd, costf, rcond, nrsing , nrobs ) ;
   fclose(fp);
}

void static tecfld(const matrix_type * W,
           const char * fname,
           const char * vname,
           int idim,
           int jdim,
           int izone)
{
   FILE *fp;

   if (izone == 1){
      fp = fopen(fname, "w");
      fprintf(fp, "TITLE = \"%s\"\n",fname);
      fprintf(fp, "VARIABLES = \"i-index\" \"j-index\" \"%s\"\n",vname);
      fprintf(fp, "ZONE F=BLOCK, T=\"Iteration %d\", I=%d, J=%d, K=1\n",izone, idim, jdim);
      {
         int ic=0;
         for (int j=0;j<jdim;j++){
            for (int i=0;i<idim;i++){
               ic=ic+1;
               fprintf(fp,"%d ",i);
               if (ic==30){
                  ic=0;
                  fprintf(fp,"\n");
               }
            }
         }
      }

      {
         int ic=0;
         for (int j=0;j<jdim;j++){
            for (int i=0;i<idim;i++){
               ic=ic+1;
               fprintf(fp,"%d ",j);
               if (ic==30){
                  ic=0;
                  fprintf(fp,"\n");
               }
            }
         }
      }
   } else {
      fp = fopen(fname, "a");
      fprintf(fp, "ZONE F=BLOCK, T=\"Iteration %d\", I=%d, J=%d, K=1, VARSHARELIST = ([1,2]=1)",izone, idim, jdim);
   }

   {
      int ic=0;
      for (int j=0;j<jdim;j++){
         for (int i=0;i<idim;i++){
            ic=ic+1;
            fprintf(fp,"%12.5e ",matrix_iget(W , i,j));
            if (ic==30){
               ic=0;
               fprintf(fp,"\n");
            }
         }
      }
   }
   fclose(fp);
}

