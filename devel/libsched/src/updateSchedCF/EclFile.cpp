#include "EclFile.h"
#include "textf.h"
//#include <afxdlgs.h>
#include <string.h>


//#include <fstream.h>
//#include "EclGrid\textf.h"
//#include <afxwin.h>
//#include <math.h>
//#include <iomanip.h>


EclFile::EclFile(char* filename,EclFile* _pParent){


//  pCurrent=new ifstream(filename,ios::nocreate);
  pCurrent=new ifstream(filename);
  pParent=_pParent;

  /*
  filePath=new char[256];

  i =strlen(filename);
  while (filename[i]!='\\')
	  i--;

  strcpy(filePath,filename);
  filePath[i+1]='\0';
*/

};

EclFile::EclFile(char* filename){

  int i;

//  pCurrent=new ifstream(filename,ios::nocreate);
  pCurrent=new ifstream(filename);
  pParent=NULL;

  filePath=new char[256];

  i =strlen(filename);
  while (filename[i]!='\\')
	  i--;

  strcpy(filePath,filename);
  filePath[i+1]='\0';

};

EclFile::~EclFile(){

   delete pCurrent;
};


bool EclFile::fileExist(char* filename){

  FILE *stream;
  bool res=true;

  if( (stream  = fopen( filename, "r" )) == NULL )
      res=false;
  
  return res;

}


EclFile* EclFile::seekKeyw(EclFile* datafil,char* keyW,bool seekOnlyMainFile){

   if (seekOnlyMainFile)
     return EclFile::_seekKeyw_mf(datafil,keyW);
   else
     return EclFile::_seekKeyw_withIncl(datafil,keyW);

}

EclFile* EclFile::_seekKeyw_mf(EclFile* datafil,char* keyW){

	char* str=new char[256];
	char* test=new char[256];
    bool found=false;
    
	int lkeyw=strlen(keyW);


    while ((!datafil->pCurrent->eof()) && (!found)){

	   datafil->pCurrent->getline(str,256);
	   
	   if (str[strlen(str)-1]=='\r')
		   str[strlen(str)-1]='\0';

	   if ((int)strlen(str)>lkeyw)
		   str[lkeyw]='\0';

	   if (strlen(str)>0)
	     str=textf::toUpper(textf::getStr(str,1));


//       if (strcmp(str,_strupr(keyW))==0)
//		   found=true;

       if (strcmp(str,keyW)==0)
		   found=true;

    };

    delete str;
	
	if (found)
		return datafil;
	else
		return NULL;


}

EclFile* EclFile::_seekKeyw_withIncl(EclFile* datafil,char* keyW){

	bool found=false;
	char* str1=new char[256];
	char* str=new char[256];

  	EclFile* inclfil;


    while ((!datafil->pCurrent->eof()) && (!found)){

	   datafil->pCurrent->getline(str,256);
	   str=textf::toUpper(str);
	   
       strcpy(str1,str);

       if (strlen(str1)>0)
	     str1=textf::getStr(str1,1);

       str1=textf::toUpper(textf::getStr(str1,1));

	   str1[strlen(keyW)]='\0';

//       if (strcmp(str1,_strupr(keyW))==0)
       if (strcmp(str1,keyW)==0)
	     found=true;

       if (strcmp(str,"INCLUDE")==0){

	     datafil->pCurrent->getline(str,256);

	     /*if (datafil->pParent==NULL){
	       pos=pos+strlen(str)+1;
		 }*/

		 if (textf::inStr(str,'\'',0) >-1 ){
			 str=textf::inside(str,'\'');
		 } else {
			 str=textf::getStr(str,1);
			 str=textf::trimL(str);
			 str=textf::trimR(str);
		 }

		 if ((str[1]!=':') && (str[1]!='\\')){
           char* str2 = new char[256];

		   strcpy(str2,datafil->filePath);
		   strcat(str2,str);
		   strcpy(str,str2);

		   delete str2;
		 }


		 if (!EclFile::fileExist(str)) {

		 	char* message=new char[256];
			sprintf(message,"Could not open the include file \n <%s> ",str);

          //  AfxMessageBox(message,MB_ICONSTOP   );
           printf("%s \n\n",message);
	      
			delete message;
			datafil->pCurrent->close();
			
			while (datafil->pParent!=NULL){
               inclfil=datafil;
			   datafil=inclfil->pParent;
			   delete inclfil;
			}

	    	delete datafil;
            delete str; 

			return false;
		 }
		 
		 inclfil=new EclFile(str,datafil);
		 datafil=inclfil;
 	     datafil->pCurrent->getline(str,256);

		 /*
		 if (datafil->pParent==NULL){
	       pos=pos+strlen(str)+1;
		 }
		 */

	   }

	   // sjekk keywords her !!!

	   if ((datafil->pCurrent->eof()) && (datafil->pParent!=NULL)) {

		   datafil->pCurrent->close();
		   inclfil=datafil;
		   datafil=inclfil->pParent;
		   delete inclfil;
	   }
	}


	delete str;

	if (found)
		return datafil;
	else
		return NULL;

}



void EclFile::nextRecord(EclFile* datafil,char* str,bool removeLineBreak){


	char* str1=new char[256];
	int pos=-1;
    int lstr=0;

	strcpy(str,"");
	
	while (pos==-1){
	  datafil->pCurrent->getline(str1,256);
	  textf::replace(str1,'\t',' ');

      while (((str1[0]=='-') && (str1[1]=='-')) || (textf::countNonSpace(str1)==0)){
	    datafil->pCurrent->getline(str1,256);
	    textf::replace(str1,'\t',' ');
	  }
    
	  pos=textf::inStr(str1,'/',0);
      
	  str1=textf::trimR(str1);
	  strcat(str1," ");

	  if (removeLineBreak){
	    
		if (pos==-1)
	      strcat(str,str1);
	    else {
          //str1[pos]='\0';
	      strcat(str,str1);
		}

	  } else {

		  //strcat(str,"\n");
		  strcat(str,str1);
          //str=textf::addStr(str,str1);
		  
		  strcat(str,"\n");
          lstr=strlen(str);
 
	  }


	};  // wend

	if (str[strlen(str)-1]=='\n')
		str[strlen(str)-1]='\0';

	//delete str1;


}

void EclFile::nextRecord(EclFile* datafil,char* str,bool removeLineBreak,bool removeComments){


	char* str1=new char[265];
	int pos=-1;

	strcpy(str,"");
	

	while (pos==-1){
	  datafil->pCurrent->getline(str1,256);
	  textf::replace(str1,'\t',' ');

	  if (removeComments){
        while (((str1[0]=='-') && (str1[1]=='-')) || (textf::countNonSpace(str1)==0)){
	      datafil->pCurrent->getline(str1,256);
	      textf::replace(str1,'\t',' ');
		}

	  } else {

		while (textf::countNonSpace(str1)==0){
	      datafil->pCurrent->getline(str1,256);
	      textf::replace(str1,'\t',' ');
		}

	  }
    
	  pos=textf::inStr(str1,'/',0);

	  if ((!removeComments) && (str1[0]=='-') && (str1[1]=='-'))
		  pos=-1;

	  if (removeLineBreak){
	    
		if (pos==-1)
	      strcat(str,str1);
	    else {
          //str1[pos]='\0';
	      strcat(str,str1);
		}

	  } else {

		  //strcat(str,"\n");
		  strcat(str,str1);
		  strcat(str,"\n");


	  }


	};  // wend

	if (str[strlen(str)-1]=='\n')
		str[strlen(str)-1]='\0';

}

void EclFile::nextLine(EclFile* datafil,char* str){


    char* str1=new char[256];
    int pos=-1;
    int lstr=0;

    strcpy(str,"");
	  
    datafil->pCurrent->getline(str,256);
    textf::replace(str1,'\t',' ');

    
}
