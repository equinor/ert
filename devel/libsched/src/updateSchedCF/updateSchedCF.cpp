#include <iostream.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
//#include <direct.h>

#include "eclgrid.h"
#include "textf.h"


#include <fstream.h>
//#include <string.h>
#include <stdio.h>
#include <ctype.h>

#include "EclFile.h"
#include "TrjFile.h"


void test(double* _kh){

  
  printf(" test foer :  %10.2f \n",*_kh);
  
  *_kh=33;  

  
  printf(" test etter:  %10.2f \n",*_kh);
}

  
int main(int argc, char** argv){

  int i,l1,l2; 
  int I,J,K;
  double S,d,KH,L;
  double cf1,cf2,diff;

  int pos,m,l;

 
 EclGrid  *grid1;
 grid1=new EclGrid();

 char* str1= new char[256];
 char* str2= new char[256];
 char* str3= new char[256];
 char* simcase=new char[256];

 
 char* substr1;
 char* substr2;
 char* cfStr;
 char* linje;
  
 char* gridFileN = new char[256];
 char* initFileN = new char[256];
 char* schedFileN = new char[256];
 char* trjFileN = new char[256];
 char* nySchedFileN  = new char[256];
 char* debugFileN  = new char[256];
 char* path  = new char[256];


 if (argc < 2){
   printf("\nScript needs one argument which is init file name \n\n");
   exit(1);
 }


 ifstream* pInitFile;
 ifstream* pDataFile;
 ofstream* pNyDataFile;

 // printf("\n%s\n\n",argv[1]);

 pInitFile=new ifstream(argv[1]);

 while (!pInitFile->eof()){

   pInitFile->getline(str1,256);

   strcpy(str2,str1);
   textf::getStr2(str2,1);
   textf::toUpper2(str2);
   textf::trimR2(str2);
   
   if (strcmp(str2,"SIMCASE")==0){
     strcpy(simcase,str1);
     textf::getStr2(simcase,2);
     // printf("Simcase: %s \n",simcase);
   }

   if (strcmp(str2,"TRJFILE")==0){
     strcpy(trjFileN,str1);
     textf::getStr2(trjFileN,2);
     // printf("Trj file: %s \n",trjFileN);
   }

   if ((strcmp(str2,"SCHEDFILE")==0) || (strcmp(str2,"SCHEDULEFILE")==0)) {
     strcpy(schedFileN,str1);
     textf::getStr2(schedFileN,2);
     // printf("sched file: %s \n",schedFileN);

     strcpy(path,schedFileN);
   
     textf::getPath(path,schedFileN);
     l1=strlen(path);
     l2=strlen(schedFileN);   

     //printf("\n\npath is |%s| %i %i \n\n",path,l1,l2);
      
     strcpy(nySchedFileN,schedFileN);
     textf::subStr(schedFileN,nySchedFileN,l1,l2-l1);


  //   textf::getFileRoot(schedFileN,nySchedFileN);
  //   printf("Ny Schedulefile: |%s| \n\n",nySchedFileN);

   }


 } 

 // oppdater denne 

 sprintf(str1,"cp %s  tmp.data",simcase);
 system (str1);

 
 pInitFile->close();
 delete pInitFile;

 pDataFile=new ifstream("tmp.data");
 pNyDataFile=new ofstream(simcase);

 while (!pDataFile->eof()){
 
   pDataFile->getline(str1,256);
   
   strcpy(str2,str1);
   textf::replace2(str2,'\'',' ');
   textf::getStr2(str2,1);
   textf::trimL2(str2);
   textf::trimR2(str2);
   
   if (strcmp(str2,schedFileN)==0){
     *pNyDataFile << " '" << nySchedFileN << "'  / \n";

   } else {
     *pNyDataFile << str1 << "\n";
    
   }
     
 
 }
 
 
 pDataFile->close();
 pNyDataFile->close();

 delete pDataFile;
 delete pNyDataFile;

 strcpy(debugFileN,"updateSchedCF.out");
 
 
/* 
 //  =============================================================================================
 //
 //  oppdater med kode, les f.eks fra en inifil ?? 
 //
  
 strcpy(gridFileN,"/project/njord/ressim/ef/2003a/tskille/updateSchedCF/NJORD_EF_HM6_38_3.GRID");
 strcpy(initFileN,"/project/njord/ressim/ef/2003a/tskille/updateSchedCF/NJORD_EF_HM6_38_3.INIT");
 
 strcpy(schedFileN,"/project/njord/ressim/ef/2003a/tskille/updateSchedCF/a8hp_test.sch");
 
 strcpy(trjFileN,"/project/njord/ressim/ef/2003a/tskille/updateSchedCF/njord_ef_test_unix.trj");
 strcpy(nySchedFileN,"/project/njord/ressim/ef/2003a/tskille/updateSchedCF/ny_test.sch");
  
  
 
 //  =============================================================================================
*/

 
 char* wellN = new char[256];
     
 Well* w1;
 Conn* c1;
 
 TrjFile* trjF1;
 trjF1=new TrjFile(trjFileN);
 
// trjF1->print("trjFile.out");

 l1=strlen(simcase);
 pos=l1;

 while ((pos>0) && (simcase[pos]!='.')){
   pos=pos-1;
 }

 if (pos>0){
   //strcpy(gridFileN,simcase);
   textf::subStr(simcase,gridFileN,0,pos);
   textf::subStr(simcase,initFileN,0,pos);
   
   strcat(gridFileN,".GRID");
   strcat(initFileN,".INIT");
   
 } else {
   strcpy(gridFileN,simcase);
   strcat(gridFileN,".GRID");
   
 } 

/* 
 printf("\nGrid File is: |%s| \n",gridFileN);
 printf("INIT File is: |%s| \n",initFileN);
*/ 
 
 
 grid1->import_grid(gridFileN);
 grid1->read_init(initFileN);

 
 EclFile* schedfil;
	
 schedfil=new EclFile(schedFileN);

 ofstream* utfil = new ofstream(nySchedFileN);
 ofstream* debug = new ofstream(debugFileN);

 sprintf(str3,"%10s %3s %3s %3s %8s %8s %8s %12s | %5s %5s %5s %5s %5s %5s %8s %8s %8s  | %8s %8s %8s\n","Well Name","I","J","K","CF","Diam","Skin","KH","kx","ky","kz","hx","hy","hz","kh","Length","multL","CF_NY","Diff CF","Diff KH");
 *debug << str3;



 while (!schedfil->pCurrent->eof()){

   EclFile::nextLine(schedfil,str1);
   
   strcpy(str2,str1);

   if ((textf::inStr(str2,"--      :")>-1) && (textf::inStr(str2,"Connection")>-1)){

     //printf(" Skal legge perfor lengde |%s \n",str2);
     strcpy(str2,str1);
     
    // printf("%s | \n",str2);
     
     // str2=&str2[10];

     strcpy(wellN,str2);
     textf::getStr2(wellN,3);
     textf::trimL2(wellN);
     textf::trimR2(wellN);
     
     //strcpy(wellN,str3);

     strcpy(str3,str1);
     textf::getStr2(str3,5);
     I=textf::toInt(str3);

     strcpy(str3,str1);
     textf::getStr2(str3,6);
     J=textf::toInt(str3);

     strcpy(str3,str1);
     textf::getStr2(str3,7);
     K=textf::toInt(str3);

     strcpy(str3,str1);
     textf::getStr2(str3,10);
     L=textf::toDouble(str3);
 
     //printf("| >%s< %i %i %i <%s> %10.2f ",wellN,I,J,K,str3,L);

     w1=NULL;
     w1=trjF1->findWell(wellN);
     
     if (w1!=NULL){
      // printf(" | fant broenn <%s> | conn %i %i %i \n",wellN,I,J,K);

       c1=NULL;
       c1=w1->findConnection(I,J,K);
//       c1=w1->findConnection_test(I,J,K);
       
       if (c1!=NULL){
          c1->setLength(L); 	
	  //printf(" | > fant conn ");
       } else{ 
       
         printf("\n\nStopper her !!\n\n");
	 printf("Skal finne %i  %i  %i  !!\n",I,J,K);
	 printf("  str1: %s \n\n",str1);
	 printf("  str2: %s \n\n",str2);
	 printf("  str3: %s \n\n",str3);
	 
	 debug->flush();
	 
         exit(1);
       }
       
       
       //printf("\n");
            
     }

	    
   }


   strcpy(str3,str2);   
   str3[7]='\0';
   textf::toUpper2(str3);
   
   
   
   if (strcmp(str3,"COMPDAT")==0){

      *utfil << "-- Connection factors (CF) recalculated by program updaetSchedCF \n\n";
      *utfil << "COMPDAT\n";
      
      utfil->flush();
      
      //printf("fant COMPDAT => ");
      
      EclFile::nextLine(schedfil,str1);
      // printf("%s\n",str1);


 //               *utfil << "**" << str1 << "\n";

      while (str1[0]!='/'){
       
        if ((str1[0]!='-') && (str1[0]!='-')){
	  
	  strcpy(str2,str1);

	  str3=textf::getItem(str2,1);
	  //printf(" %s  ",str3);

	  str3=textf::replace(str3,'\'',' ');
	  str3=textf::trimL(str3);
	  str3=textf::trimR(str3);
          
	  strcpy(wellN,str3);
	     
	  strcpy(str2,str1);
	  str3=textf::getItem(str2,2);
          I=textf::toInt(str3);

	  strcpy(str2,str1);
	  str3=textf::getItem(str2,3);
          J=textf::toInt(str3);

	  strcpy(str2,str1);
	  str3=textf::getItem(str2,4);
          K=textf::toInt(str3);

	  strcpy(str2,str1);
	  str3=textf::getItem(str2,8);
          cf1=textf::toDouble(str3);

          // printf("cf1: %10.4f \n",cf1);
	    
	  strcpy(str2,str1);
	  str3=textf::getItem(str2,9);
          d=textf::toDouble(str3);

          // printf("d: %10.4f \n",d);

	  strcpy(str2,str1);
	  str3=textf::getItem(str2,11);
	  
          if (strcmp(str3,"*")!=0)
            S=textf::toDouble(str3);
          else
	    S=0.0;
	    
         // printf("S: %10.4f \n",S);
   
	  strcpy(str2,str1);
	  str3=textf::getItem(str2,10);
          KH=textf::toDouble(str3);

          //printf("KH: %10.4f \n",KH);

          sprintf(str3,"%10s %3i %3i %3i %8.3f %8.3f %8.3f %12.3f ",wellN,I,J,K,cf1,d,S,KH);
          *debug << str3;
          
	  debug->flush();
	  
          w1=NULL;
	  	  
          w1=trjF1->findWell(wellN);

          if (w1==NULL){
	    printf("\n\nError !! \nCould not find well <%s> in Trajectorifile \n\n",wellN);
	    exit(1);
	  }  
         
          c1 = NULL;
          c1=w1->findConnection(I,J,K);
          
          if (c1==NULL){
	    printf("\n\nError !! \nCould not find connection <%2i %2i %2i> in well <%s> in Trajectorifile \n\n",I,J,K,wellN);
	    exit(1);
	  }  

        //  cf2=c1->calcConn(grid1,S,d,debug);	 
          
	  double kh=999;
	  double* pkh;
	  double diffKh;
	  
	  pkh=&kh;

	  
	  cf2=c1->calcConn(grid1,S,d,debug,pkh);	 

	  // exit(1);
	  
	  // cf2=1.0;
	  
	  diff = fabs(cf1-cf2)/cf1*100;
	  diffKh = fabs(KH-kh)/KH*100;
	  
          sprintf(str3," | %8.3f %8.2f %8.2f ",cf2,diff,diffKh);
	   
          *debug << str3 << "\n";
	  
	  
	  strcpy(str2,str1);
	
	  textf::getItem2(str2,8);
          pos=textf::inStr(str1,str2);
          l=strlen(str2);
	  

          substr1=new char[256];
          substr2=new char[256];
          cfStr=new char[256];
          linje=new char[256];

          sprintf(cfStr," %6.2f ",cf2); 
          textf::subStr(str1,substr1,0,pos);
          textf::trimR(substr1);  
  
          textf::subStr(str1,substr2,pos+l,strlen(str1)-pos-l-1);
     
	  sprintf(linje,"%s %10.3f %s ",substr1,cf2,substr2);

          *utfil <<  linje << "\n";
 
	  
          delete substr1;
          delete substr2;
          delete cfStr;
          delete linje;
 
	  
	 	  
       
        } else  {
       
          // kommentar linje
          *utfil << str1 << "\n";
          utfil->flush();
	      
        } 

        EclFile::nextLine(schedfil,str1);
       
      }

      *utfil << "/" << "\n";

      // exit(1);
      
   } else {
   
     *utfil << str1 << "\n";

   }
   
 } 




 debug->close();
 utfil->flush();
 
 delete utfil;
 delete schedfil;
 
 system ("rm tmp.data");
 
 return 0; 

}
  
