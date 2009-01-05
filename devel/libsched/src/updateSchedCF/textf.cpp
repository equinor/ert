#include "textf.h"
#include <string.h>
#include <stdio.h>
//#include <afxdlgs.h>

#include <iostream.h>

textf::textf(){

};

textf::~textf(){

};


int textf::toInt(char* _str){

  int value;
  
  
  if (!sscanf(_str,"%d",&value)){
    cout << "\nError !!\nCan't convert " << _str 
		<< " to integer \n\n";
	exit(1);
  }
	
	
  return value;

}

double textf::toDouble(char* _str){

  double value;
  
  if (!sscanf(_str,"%lf",&value)){
    cout << "\nError !!\nCan't convert " << _str 
		<< " to double \n\n";
	exit(1);
  }
	
  return value;

}

void textf::nextStr(char* _str,ifstream* stream){

  char t;
  int i;

  do {
    t=stream->get();
  } while ( ((t=='\n') || (t==' ') || (t=='\t') || (t=='\'')) && (!stream->eof()));
  
  i=0;
  _str[0]=t;

  while ((t!=' ') && (t!='\t') && (t!='\n')  && (t!='\'') && (! stream->eof())){
    i++;
    t=stream->get();
    _str[i]=t;
  };

  if (t!='\'')
    t=stream->get();

  _str[i]='\0';


};


void textf::nextStr(char* _str,ifstream* stream,char c){

  char t;
  int i;

  t=stream->get();
  while ((t!=c) && (!stream->eof())){
    t=stream->get();
  } 
  
  i=0;
  t=stream->get();
  _str[0]=t;

  while ((t!=c) && (! stream->eof())){
    i++;
    t=stream->get();
    _str[i]=t;
  };


  _str[i]='\0';


};

void textf::nextLine(char* _str,ifstream* stream){

  int i=0;
  char t;

  // AfxMessageBox("Trodde jeg ikke skulle bruke denne !!!");
  exit(0);

/*
  char* message=new char[100];
  
  for (i=0;i<40;i++)
    _str[i]=stream->get();

  for (int j=0;j<100;j++)
	if ((int)_str[10]==j){
	  sprintf(message,"number is: %i ",j);
	  AfxMessageBox(message);
	}

	exit(0);
*/  

  while ((t!='\n') && ((int)t!=10) && (!stream->eof())){

    t=stream->get();
    _str[i]=t;
	i++;

  };

  if ((t=='\n') || ((int)t==10))
    _str[i-1]='\0';
  else
    _str[i]='\0';

}

int textf::countStr(char* str){

  int ant,strl,i;
  char c;

  if (strlen(str)>0){
    
	ant=0;
	i=0;
	strl=strlen(str);

	do {
      c=str[i];

      while (((c==' ') || (c=='\t')) && (i<strl)){
        i++;
        c=str[i];
	  }

	  if (i<strl){
        ant++;
        
		while ((c!=' ') && (c!='\t') && (i<strl)){
          i++;
          c=str[i];
		}
	  }

	} while (i<strl);

  
  } else {
	ant= -1;
  }


  return ant;

}

double textf::getDouble(char* str,int n){

  int ant,strl,i,j;
  char c;
  char* str1 = new char[256];

  if (strlen(str)>0){
    
	ant=0;
	i=0;
	strl=strlen(str);

	do {
      c=str[i];

      while (((c==' ') || (c=='\t')) && (i<strl)){
        i++;
        c=str[i];
	  }

	  if (i<strl){
        ant++;
        j=0;
		str1[j]=c;
        j++;
		
		while ((c!=' ') && (c!='\t') && (i<strl)){
          i++;
          c=str[i];
		  str1[j]=c;
		  j++;
		}

		str1[j]='\0';

	  }

	} while (ant<n);
  }

  return toDouble(str1);

};

char* textf::trimL(char* str){

   int l=strlen(str);
   int i=0;
   
   while ((i<l) && (str[i]==' '))
	   i++;

   if (i>0){
     for (int j=0;j<l;j++)
	   str[j]=str[j+i];
	 str[l]='\0';
   }
   

   return str;

}

char* textf::trimR(char* str){

   int l=strlen(str)-1;
   int i=0;
   
   while ((l>0) && (str[l]==' '))
	   l--;

   str[l+1]='\0';


   return str;

}

void textf::trimL2(char* str){

   int l=strlen(str);
   int i=0;
   
   while ((i<l) && (str[i]==' '))
	   i++;

   if (i>0){
     for (int j=0;j<l;j++)
	   str[j]=str[j+i];
	 str[l]='\0';
   }
   

   // return str;

}

void textf::trimR2(char* str){

   int l=strlen(str)-1;
   int i=0;
   
   while ((l>0) && (str[l]==' '))
	   l--;

   str[l+1]='\0';


   // return str;

}

char* textf::toUpper(char* str){

   int l=strlen(str);
   int i=0;
   
   for (i=0;i<l;i++)
     str[i]=toupper(str[i]);

   return str;

}

void textf::toUpper2(char* str){

   int l=strlen(str);
   int i=0;
   
   for (i=0;i<l;i++)
     str[i]=toupper(str[i]);

   // return str;

}


char* textf::replace(char* str,char r,char n){

   int i =0;

   for (i=0;i<(int)strlen(str);i++)
	   if (str[i]==r){
		 str[i]=n;
	   }

   return str;

}

void textf::replace2(char* str,char r,char n){

   int i =0;

   for (i=0;i<(int)strlen(str);i++)
	   if (str[i]==r){
		 str[i]=n;
	   }

  // return str;

}

char* textf::inside(char* str,char c){

	int i=0;  int j=0;
    int l=strlen(str);
    
	for (i=0;i<l;i++)
	  if (str[i]==c)
        j++;
		 
	if (j>=2){
      
      i=0;
	  while (str[i]!=c)
		  i++;

	  j=i+1;
      while (str[j]!=c)
		  j++;

	  for (int k=0;k<j-i-1;k++)
        str[k]=str[i+k+1];

	  str[j-i-1]='\0';
	} 

	return str;

}


int textf::inStr(char* str,char c,int pos){
  
   int i=0;
   int res=-1;

   if (pos < (int)strlen(str))
     for (i=pos;i<(int)strlen(str);i++)
	if (str[i]==c)
	   res=i;

 

    return res;

}

int textf::inStr(char* str1,char* str2){
  
   int i,j;
   int pos;
   int l1,l2;
   int l;
   
   char* tmp=new char[1000];
   
   
   l1=strlen(str1);
   l2=strlen(str2);
   
   pos=-1;
   
   if (strcmp(str1,str2)==0)
     pos=0;
   
   if (l1>l2){
     l=l1-l2+1;
     for (i=0;i<l;i++){
        for (j=0;j<l2;j++)
          tmp[j]=str1[i+j];
	tmp[l2]='\0';
	  
        if (strcmp(str2,tmp)==0){
	  pos=i;
	  i=l-1;
	}  
     }
   
   }

   delete tmp;
   
    return pos;

}


char* textf::getStr(char* str,int n){

  int ant,strl,i,j;
  char c;

  //char* str=new char[256];
  //strcpy(str,str1);
  
  j=-1;

  if (strlen(str)>0){
    
	ant=0;
	i=0;
	strl=strlen(str);

	do {
      c=str[i];

      while (((c==' ') || (c=='\t')) && (i<strl)){
        i++;
        c=str[i];
	  }

	  if (i<strl){
        ant++;
        j=0;
		str[j]=c;
        j++;
		
		while ((c!=' ') && (c!='\t') && (i<strl)){
          i++;
          c=str[i];
		  str[j]=c;
		  j++;
		}

		//str[j]='\0';

	  } else {
	    ant=n;
		str[0]='\0';

	  }


	} while (ant<n);

	if (j>-1)
	  str[j]='\0';

  }

  return str;

};

void textf::getStr2(char* str,int n){

  int ant,strl,i,j;
  char c;

  //char* str=new char[256];
  //strcpy(str,str1);
  
  j=-1;

  if (strlen(str)>0){
    
	ant=0;
	i=0;
	strl=strlen(str);

	do {
      c=str[i];

      while (((c==' ') || (c=='\t')) && (i<strl)){
        i++;
        c=str[i];
	  }

	  if (i<strl){
        ant++;
        j=0;
		str[j]=c;
        j++;
		
		while ((c!=' ') && (c!='\t') && (i<strl)){
          i++;
          c=str[i];
		  str[j]=c;
		  j++;
		}

		//str[j]='\0';

	  } else {
	    ant=n;
		str[0]='\0';

	  }


	} while (ant<n);

	if (j>-1)
	  str[j]='\0';

  }

  // return str;

};

/*
char* textf::cstr2str(char* str,CString cstr){

   int i;

   for (i = 0;i<cstr.GetLength();i++)
   	  str[i]=cstr.GetAt(i);

   str[cstr.GetLength()]='\0';
 
   return str;

}
*/

void textf::getPath(char* path, char *str)
{

   int i=strlen(str);

   while ((i>0) && (str[i]!='\\') && (str[i]!='/'))
	   i--;

   if (i!=0){
	 strcpy(path,str);
	 path[i+1]='\0';
   }


}

void textf::getFileRoot(char *str, char *root)
{
  char* str1=new char[256];
  int l1,l2,i;
   
  l1=strlen(str);
  getPath(str1,str);
  l2=strlen(str1);

  for (i=l2;i<l1;i++)
    root[i-l2]=str[i];

  root[l1-l2]='\0';
  
  l1=strlen(root);
  l2=-1;
  
  for (i=0;i<l1;i++)
	if (root[i]=='.')
	  l2=i;

  if (l2>-1)
    root[l2]='\0';
		  

  delete str1;

}

int textf::countNonSpace(char* str){

	int n;

	n=0;
	for (int i=0;i<(int)strlen(str);i++)
	  if (str[i]!=' ')
	    n++;

    return n;

}

char* textf::getItem(char* str,int n){


    char* str1 = new char[256];
    char* str2 = new char[256];
    char* str3 = new char[256];

	int a,b,i,j,pos;

    strcpy(str1,"");

	a=countStr(str);

	for (i=0;i<a;i++){
	 strcpy(str2,str);
     textf::getStr(str2,i+1);
	 str2=textf::trimR(str2);
	 pos=textf::inStr(str2,'*',0);


	 if (pos==-1){
	   strcat(str1,str2);
	   strcat(str1," ");
	 } else {

	   str2=textf::trimR(str2);

       for (j=0;j<pos;j++)
	     str3[j]=str2[j];
		
	   str3[pos]='\0';
       if (strlen(str3)>0)
	     b=textf::toInt(str3);
       else
	     b=1;

       if ((strlen(str2)-pos)>1){
		   for (j=pos+1;j<(int)strlen(str2);j++)
			   str3[j-pos-1]=str2[j];
		   str3[strlen(str2)-pos-1]='\0';

	   } else
		   strcpy(str3,"*");

       for (j=0;j<b;j++){
	     strcat(str1,str3);
	     strcat(str1," ");
	   }
	 } // if 
	}  // for


	if (n>countStr(str1))
	  strcpy(str,"*");
	  else {
      textf::getStr(str1,n);
	  strcpy(str,str1);
	}

	delete str1;
	delete str2;
	delete str3;

	if (str[0]=='/')	 
	  strcpy(str,"*");

	str=textf::trimL(str);
	str=textf::trimR(str);
	
	return str;

}


void textf::getItem2(char* str,int n){


    char* str1 = new char[256];
    char* str2 = new char[256];
    char* str3 = new char[256];

	int a,b,i,j,pos;

    strcpy(str1,"");

	a=countStr(str);

	for (i=0;i<a;i++){
	 strcpy(str2,str);
     textf::getStr(str2,i+1);
	 str2=textf::trimR(str2);
	 pos=textf::inStr(str2,'*',0);


	 if (pos==-1){
	   strcat(str1,str2);
	   strcat(str1," ");
	 } else {

	   str2=textf::trimR(str2);

       for (j=0;j<pos;j++)
	     str3[j]=str2[j];
		
	   str3[pos]='\0';
       if (strlen(str3)>0)
	     b=textf::toInt(str3);
       else
	     b=1;

       if ((strlen(str2)-pos)>1){
		   for (j=pos+1;j<(int)strlen(str2);j++)
			   str3[j-pos-1]=str2[j];
		   str3[strlen(str2)-pos-1]='\0';

	   } else
		   strcpy(str3,"*");

       for (j=0;j<b;j++){
	     strcat(str1,str3);
	     strcat(str1," ");
	   }
	 } // if 
	}  // for


	if (n>countStr(str1))
	  strcpy(str,"*");
	  else {
      textf::getStr(str1,n);
	  strcpy(str,str1);
	}

	delete str1;
	delete str2;
	delete str3;

	if (str[0]=='/')	 
	  strcpy(str,"*");

	str=textf::trimL(str);
	str=textf::trimR(str);
	
//	return str;

}



bool textf::nullRecord(char *str)
{

	int pos;
	char* str1=new char[strlen(str)];

	strcpy(str1,str);

	pos=textf::inStr(str,'/',0);
	
	if (pos > -1) {
	  str1[pos]='\0';
	  pos=strlen(str1);
	}

	if (pos==0)
		return true;
	else
		return false;

	delete str1;

}


char* textf::addSpaceR(char* str,int n){

	int i;
	int l;

	l=strlen(str);

	if (l<n){
	  for (i=l;i<n;i++)
		  str[i]=' ';

	  str[n]='\0';
	}

	return str;

}


char* textf::addStr(char* addTo,char* str){

  int l1,l2,i;

  l1=strlen(addTo);
  l2=strlen(str);

  for (i=l1;i<l1+l2;i++)
	addTo[i]=str[i-l1];   

  addTo[l1+l2]='\0';


  return addTo;

}

void textf::subStr(char* source,char* dest,int start,int ant){

 int i,n;

 n=0;
 for (i=start;i<(start+ant);i++){
   dest[n]=source[i];
   n++;
 }

 dest[n]='\0';

}
