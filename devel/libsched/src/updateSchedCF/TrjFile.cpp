#include "TrjFile.h"
#include "textf.h"
//#include <afxdlgs.h>
#include <string.h>
#include "eclgrid.h"
#include "geom.h"

//#include <fstream.h>
//#include "EclGrid\textf.h"
//#include <afxwin.h>
//#include <math.h>
//#include <iomanip.h>

//////////////////////////////////
// Conn member functions 
//////////////////////////////////

Conn::Conn(){

  I=-1;
  J=-1;   
  K=-1;
 
  kx=0.0;
  ky=0.0;
  kz=0.0;
  ntg=0.0;
 
  entry_md=0.0;
  entry_x=0.0;
  entry_y=0.0;
  entry_z=0.0;
       
  exit_md=0.0;
  exit_x=0.0;
  exit_y=0.0;
  exit_z=0.0;

  L=0.0;
    
}



Conn::~Conn(){

  // slett data som er generet ??
  // delete pCurrent;

};


void Conn::addLength(double _L){

  L+=_L;
  
}


void Conn::setLength(double _L){

  L=_L;
  
}

void Conn::print(){

  printf("%2i %2i %2i | %15.1f %5.1f %5.1f %4.2f | %5.1f  %9.1f %9.1f %5.1f  |  %5.1f  %9.1f %9.1f %5.1f   \n",I,J,K,kx,ky,kz,ntg,entry_md,entry_x,entry_y,entry_z,exit_md,exit_x,exit_y, exit_z);

};

double Conn::calcConn(EclGrid* simgrid,double S,double d,ofstream* debug ){

  
  Cell* c1;
  double*** kxTab=NULL;
  double*** kyTab=NULL;
  double*** kzTab=NULL;
  double*** ntgTab=NULL;
  double kx,ky,kz,ntg,dx,dy,dz;

  double khx,khy,khz,kh;  
  double hx,hy,hz;
  double rox,roy,roz;
  double rw,c;
  double Tx,Ty,Tz,T;
  double pi;
  double multL;         
  int i;
  
  double x1,y1,z1;
  double x2,y2,z2;
  char* str3=new char[256];
    
  c1=&simgrid->cell[I-1][J-1][K-1];
  
  x1=(c1->n[0].x+c1->n[2].x+c1->n[4].x+c1->n[6].x)/4.0;
  y1=(c1->n[0].y+c1->n[2].y+c1->n[4].y+c1->n[6].y)/4.0;
  z1=(c1->n[0].z+c1->n[2].z+c1->n[4].z+c1->n[6].z)/4.0;
  
  x2=(c1->n[1].x+c1->n[3].x+c1->n[5].x+c1->n[7].x)/4.0;
  y2=(c1->n[1].y+c1->n[3].y+c1->n[5].y+c1->n[7].y)/4.0;
  z2=(c1->n[1].z+c1->n[3].z+c1->n[5].z+c1->n[7].z)/4.0;
  
  vector3d i1(x2-x1,y2-y1,z2-z1);
  
  //i1.print();
  //printf("\n");


  x1=(c1->n[2].x+c1->n[3].x+c1->n[6].x+c1->n[7].x)/4.0;
  y1=(c1->n[2].y+c1->n[3].y+c1->n[6].y+c1->n[7].y)/4.0;
  z1=(c1->n[2].z+c1->n[3].z+c1->n[6].z+c1->n[7].z)/4.0;
  
  x2=(c1->n[0].x+c1->n[1].x+c1->n[4].x+c1->n[5].x)/4.0;
  y2=(c1->n[0].y+c1->n[1].y+c1->n[4].y+c1->n[5].y)/4.0;
  z2=(c1->n[0].z+c1->n[1].z+c1->n[4].z+c1->n[5].z)/4.0;

  dx=c1->calc_dx();
  dy=c1->calc_dy();
  dz=c1->calc_dz();
  
  vector3d j1(x2-x1,y2-y1,z2-z1);

  //j1.print();
  //printf("\n");
  
  vector3d i1xj1 = vector3d::cross(&i1,&j1);
  
  
  //i1xj1.print();
  //printf("\n");

  vector3d z=i1xj1.normalize();

  //printf("\nZ-vector: ");
  //z.print();
  //printf("\n");

  vector3d j1xz = vector3d::cross(&j1,&z);


  //j1xz.print();
  //printf("\n");

  vector3d i2 = i1 + j1xz;
  i2=i2.normalize();   

  //i2.print();
  //printf("\n");
  
  vector3d i1xz = vector3d::cross(&i1,&z);
  //i1xz.print();
  //printf("\n");


  vector3d j2 = j1 - i1xz;
  j2=j2.normalize();   

  //j2.print();
  //printf("\n");

  double** A = new double *[3];
  
  for (i=0;i<3;i++)
    A[i]=new double[3];  

  A[0][0]=i2.x;
  A[0][1]=i2.y;
  A[0][2]=i2.z;
       
  A[1][0]=j2.x;
  A[1][1]=j2.y;
  A[1][2]=j2.z;
  
  A[2][0]=z.x;
  A[2][1]=z.y;
  A[2][2]=z.z;

  vector3d p(exit_x-entry_x,exit_y-entry_y,exit_z-entry_z);

  vector3d x = vector3d::GaussEli(A,&p);

  hx=x.x;
  hy=x.y;
  hz=x.z;
  
  multL=1.0;
  
//  if (L>0)
//    multL=L/(pow(pow(p.x,2)+pow(p.y,2)+pow(p.z,2),0.5));

  if (L>0)
    multL=L/(exit_md-entry_md);
              
  
  kxTab=simgrid->getInitParam("PERMX   ");
  kyTab=simgrid->getInitParam("PERMY   ");
  kzTab=simgrid->getInitParam("PERMZ   ");
  ntgTab=simgrid->getInitParam("NTG     ");

  kx=0;
  ky=0;
  kz=0;
  ntg=0;
  
  if (kxTab!=NULL)
    kx=kxTab[I-1][J-1][K-1];

  if (kyTab!=NULL)
    ky=kyTab[I-1][J-1][K-1];

  if (kzTab!=NULL)
    kz=kzTab[I-1][J-1][K-1];

  if (ntgTab!=NULL)
    ntg=ntgTab[I-1][J-1][K-1];

  
  khx=hx*pow(ky*kz,0.5)*multL;
  khy=hy*pow(kx*kz,0.5)*multL;
  khz=hz*pow(kx*ky,0.5)*multL;

  kh=pow(pow(khx,2)+pow(khy,2)+pow(khz,2),0.5);

  
  rox=0.28*pow(pow(dz,2)*pow(ky/kz,0.5)+pow(dy,2)*pow(kz/ky,0.5),0.5)/(pow(ky/kz,0.25)+pow(kz/ky,0.25));
  roy=0.28*pow(pow(dz,2)*pow(kx/kz,0.5)+pow(dx,2)*pow(kz/kx,0.5),0.5)/(pow(kx/kz,0.25)+pow(kz/kx,0.25));
  roz=0.28*pow(pow(dy,2)*pow(kx/ky,0.5)+pow(dx,2)*pow(ky/kx,0.5),0.5)/(pow(kx/ky,0.25)+pow(ky/kx,0.25));


  rw=d/2.0;
  c=0.008527;    
  pi=3.141592654;
  S=0;
  
  Tx=c*2*pi*khx/(log(rox/rw)+S);
  Ty=c*2*pi*khy/(log(roy/rw)+S);
  Tz=c*2*pi*khz/(log(roz/rw)+S);

  T=pow(pow(Tx,2)+pow(Ty,2)+pow(Tz,2),0.5);
  
  sprintf(str3,"| %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %8.1f %8.2f %8.2f ",kx,ky,kz,hx,hy,hz,kh,L,multL*100);  
  *debug << str3;

// midlertidig !!
// return kh;

  return T;
  
}


double Conn::calcConn(EclGrid* simgrid,double S,double d,ofstream* debug,double* _kh){

  
  Cell* c1;
  double*** kxTab=NULL;
  double*** kyTab=NULL;
  double*** kzTab=NULL;
  double*** ntgTab=NULL;
  double kx,ky,kz,ntg,dx,dy,dz;

  double khx,khy,khz,kh;  
  double hx,hy,hz;
  double rox,roy,roz;
  double rw,c;
  double Tx,Ty,Tz,T;
  double pi;
  double multL;         
  int i;
  
  double x1,y1,z1;
  double x2,y2,z2;
  char* str3=new char[256];
    
  c1=&simgrid->cell[I-1][J-1][K-1];

  //c1->print();
  
  
  x1=(c1->n[0].x+c1->n[2].x+c1->n[4].x+c1->n[6].x)/4.0;
  y1=(c1->n[0].y+c1->n[2].y+c1->n[4].y+c1->n[6].y)/4.0;
  z1=(c1->n[0].z+c1->n[2].z+c1->n[4].z+c1->n[6].z)/4.0;
  
  x2=(c1->n[1].x+c1->n[3].x+c1->n[5].x+c1->n[7].x)/4.0;
  y2=(c1->n[1].y+c1->n[3].y+c1->n[5].y+c1->n[7].y)/4.0;
  z2=(c1->n[1].z+c1->n[3].z+c1->n[5].z+c1->n[7].z)/4.0;
  
  vector3d i1(x2-x1,y2-y1,z2-z1);
  
  //i1.print();
  //printf("\n");

/*
  x1=(c1->n[2].x+c1->n[3].x+c1->n[6].x+c1->n[7].x)/4.0;
  y1=(c1->n[2].y+c1->n[3].y+c1->n[6].y+c1->n[7].y)/4.0;
  z1=(c1->n[2].z+c1->n[3].z+c1->n[6].z+c1->n[7].z)/4.0;
  
  x2=(c1->n[0].x+c1->n[1].x+c1->n[4].x+c1->n[5].x)/4.0;
  y2=(c1->n[0].y+c1->n[1].y+c1->n[4].y+c1->n[5].y)/4.0;
  z2=(c1->n[0].z+c1->n[1].z+c1->n[4].z+c1->n[5].z)/4.0;
*/


  x1=(c1->n[0].x+c1->n[1].x+c1->n[4].x+c1->n[5].x)/4.0;
  y1=(c1->n[0].y+c1->n[1].y+c1->n[4].y+c1->n[5].y)/4.0;
  z1=(c1->n[0].z+c1->n[1].z+c1->n[4].z+c1->n[5].z)/4.0;

  x2=(c1->n[2].x+c1->n[3].x+c1->n[6].x+c1->n[7].x)/4.0;
  y2=(c1->n[2].y+c1->n[3].y+c1->n[6].y+c1->n[7].y)/4.0;
  z2=(c1->n[2].z+c1->n[3].z+c1->n[6].z+c1->n[7].z)/4.0;
  

  dx=c1->calc_dx();
  dy=c1->calc_dy();
  dz=c1->calc_dz();
  
  vector3d j1(x2-x1,y2-y1,z2-z1);

  //j1.print();
  //printf("\n");
  
  vector3d i1xj1 = vector3d::cross(&i1,&j1);
  
  
  //i1xj1.print();
  //printf("\n");

  vector3d z=i1xj1.normalize();

  //printf("\nZ-vector: ");
  //z.print();
  //printf("\n");

  vector3d j1xz = vector3d::cross(&j1,&z);


  //j1xz.print();
  //printf("\n");

  vector3d i2 = i1 + j1xz;
  i2=i2.normalize();   

  //i2.print();
  //printf("\n");
  
  vector3d i1xz = vector3d::cross(&i1,&z);
  //i1xz.print();
  //printf("\n");


  vector3d j2 = j1 - i1xz;
  j2=j2.normalize();   

  //j2.print();
  //printf("\n");

  // sjekk ortogonaliltet skalarprodukt = 0

  double l1=fabs(i2.x*j2.x+i2.y*j2.y+i2.z*j2.z);
  double l2=fabs(i2.x*z.x+i2.y*z.y+i2.z*z.z);
  double l3=fabs(j2.x*z.x+j2.y*z.y+j2.z*z.z);
  
  double sl=l1+l2+l3;
   
  // printf("%3i %3i %3i   %10.3e %10.3e %10.3e  => %10.3e  \n",I,J,K,l1,l2,l3,sl);

  if (sl>1e-10){
    printf("\nError found in ortogonality Program stopped \n");
    printf("Error is %10.3e which is larger than %10.3e \n\n",sl,1e-10);
    exit(1);
    
  }   


  double** A = new double *[3];
  
  for (i=0;i<3;i++)
    A[i]=new double[3];  

  A[0][0]=i2.x;
  A[0][1]=i2.y;
  A[0][2]=i2.z;
       
  A[1][0]=j2.x;
  A[1][1]=j2.y;
  A[1][2]=j2.z;
  
  A[2][0]=z.x;
  A[2][1]=z.y;
  A[2][2]=z.z;

  vector3d p(exit_x-entry_x,exit_y-entry_y,exit_z-entry_z);

/*
  printf("\nA foer eliminasjon: \n");
  
  for (i=0;i<3;i++)
    printf(" %10.5f %10.5f  %10.5f  \n",A[0][i],A[1][i],A[2][i]);
  printf("\n\n"); 
*/

  vector3d x = vector3d::GaussEli(A,&p);

  A[0][0]=i2.x;  A[0][1]=i2.y;  A[0][2]=i2.z;
  A[1][0]=j2.x;  A[1][1]=j2.y;  A[1][2]=j2.z;  
  A[2][0]=z.x;   A[2][1]=z.y;   A[2][2]=z.z;

/*
  printf("\nA etter eliminasjon: \n");
  
  for (i=0;i<3;i++)
    printf(" %10.5f %10.5f  %10.5f  \n",A[0][i],A[1][i],A[2][i]);
  printf("\n\n"); 
*/

  hx=x.x;
  hy=x.y;
  hz=x.z;
  
  // sjekk løsning av lineære ligninger !!
  
  l1=fabs(p.x-(A[0][0]*hx+A[1][0]*hy+A[2][0]*hz));
  l2=fabs(p.y-(A[0][1]*hx+A[1][1]*hy+A[2][1]*hz));
  l3=fabs(p.z-(A[0][2]*hx+A[1][2]*hy+A[2][2]*hz));
  sl=l1+l2+l3;
  
  // printf("%3i %3i %3i   %10.3e %10.3e %10.3e  => %10.3e  \n",I,J,K,l1,l2,l3,sl);
  
  if (sl>1e-10){
    printf("\nError solving linear equations. Program stopped \n");
    printf("Error is %10.3e which is larger than %10.3e \n\n",sl,1e-10);
    exit(1);
    
  }   
  
  
  multL=1.0;
  
//  if (L>0)
//    multL=L/(pow(pow(p.x,2)+pow(p.y,2)+pow(p.z,2),0.5));

  if (L>0)
    multL=L/(exit_md-entry_md);
              
  
  kxTab=simgrid->getInitParam("PERMX   ");
  kyTab=simgrid->getInitParam("PERMY   ");
  kzTab=simgrid->getInitParam("PERMZ   ");
  ntgTab=simgrid->getInitParam("NTG     ");

  kx=0;
  ky=0;
  kz=0;
  ntg=0;
  
  if (kxTab!=NULL)
    kx=kxTab[I-1][J-1][K-1];

  if (kyTab!=NULL)
    ky=kyTab[I-1][J-1][K-1];

  if (kzTab!=NULL)
    kz=kzTab[I-1][J-1][K-1];

  if (ntgTab!=NULL)
    ntg=ntgTab[I-1][J-1][K-1];

  
  khx=hx*pow(ky*kz,0.5)*multL;
  khy=hy*pow(kx*kz,0.5)*multL;
  khz=hz*pow(kx*ky,0.5)*multL;

  kh=pow(pow(khx,2)+pow(khy,2)+pow(khz,2),0.5);
  *_kh=kh;
  
  
  rox=0.28*pow(pow(dz,2)*pow(ky/kz,0.5)+pow(dy,2)*pow(kz/ky,0.5),0.5)/(pow(ky/kz,0.25)+pow(kz/ky,0.25));
  roy=0.28*pow(pow(dz,2)*pow(kx/kz,0.5)+pow(dx,2)*pow(kz/kx,0.5),0.5)/(pow(kx/kz,0.25)+pow(kz/kx,0.25));
  roz=0.28*pow(pow(dy,2)*pow(kx/ky,0.5)+pow(dx,2)*pow(ky/kx,0.5),0.5)/(pow(kx/ky,0.25)+pow(ky/kx,0.25));


  rw=d/2.0;
  c=0.008527;    
  pi=3.141592654;
  S=0;
  
  Tx=c*2*pi*khx/(log(rox/rw)+S);
  Ty=c*2*pi*khy/(log(roy/rw)+S);
  Tz=c*2*pi*khz/(log(roz/rw)+S);

  T=pow(pow(Tx,2)+pow(Ty,2)+pow(Tz,2),0.5);
 
//  printf("\n\nCF = %10.4f  KH=  %10.3f \n\n",T,kh);
//  exit(1);
 
  
  sprintf(str3,"| %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %8.1f %8.2f %8.2f ",kx,ky,kz,hx,hy,hz,kh,L,multL*100);  
  *debug << str3;

// midlertidig !!
// return kh;

  return T;
  
}


//////////////////////////////////
// Well member functions 
//////////////////////////////////

Well::Well(){
  
  nConn=0;
  
}

Well::Well(char* _wellname){
  
  wellName=new char[256];
  strcpy(wellName,_wellname);

  nConn=0;
  
}

void Well::addConnection(int _I,int _J,int _K,double _kx,double _ky,double _kz,double _ntg,double _entry_md,double _entry_x,double _entry_y,double _entry_z,double _exit_md,double _exit_x,double _exit_y,double _exit_z){
  
  connTab[nConn]=new Conn();

  connTab[nConn]->I=_I;
  connTab[nConn]->J=_J;
  connTab[nConn]->K=_K;

  connTab[nConn]->kx=_kx;
  connTab[nConn]->ky=_ky;
  connTab[nConn]->kz=_kz;
  connTab[nConn]->ntg=_ntg;
  
  connTab[nConn]->ntg=_ntg;

  connTab[nConn]->entry_md=_entry_md;
  connTab[nConn]->entry_x =_entry_x;
  connTab[nConn]->entry_y =_entry_y;
  connTab[nConn]->entry_z =_entry_z;

  connTab[nConn]->exit_md=_exit_md;
  connTab[nConn]->exit_x =_exit_x;
  connTab[nConn]->exit_y =_exit_y;
  connTab[nConn]->exit_z =_exit_z;


  nConn++;
  
}

Conn* Well::findConnection(int _I,int _J,int _K){

  
  Conn* c1=NULL;
  int i;

  
  for (i=0;i<nConn;i++)
    if ((connTab[i]->I==_I) && (connTab[i]->J==_J) && (connTab[i]->K==_K))
      c1=connTab[i];
        
  return c1;

}


Conn* Well::findConnection_test(int _I,int _J,int _K){

  
  Conn* c1=NULL;
  int i;
  
  printf("Skal finne %i %i %i ",_I,_J,_K);
  
  for (i=0;i<nConn;i++)
    if ((connTab[i]->I==_I) && (connTab[i]->J==_J) && (connTab[i]->K==_K))
      c1=connTab[i];
  
  if (c1!=NULL)
    printf("FANT !");
  else {
    printf("\n\nFANT IKKE! \n");
    
    for (i=0;i<nConn;i++){
      printf(" %i vs %i and %i vs %i and  %i vs %i \n",connTab[i]->I,_I,connTab[i]->J,_J,connTab[i]->K,_K);
      
      if ((connTab[i]->I==_I) && (connTab[i]->J==_J) && (connTab[i]->K==_K))
        c1=connTab[i];
    }
    
    exit(1);
    
  }
        
  return c1;
  
  

}

Well::~Well(){

  // slett data som er generet ??
  // delete pCurrent;

};

void Well::printConn(){

  int i;

  printf("Antall connections:  %i \n\n",nConn);

  for (i=0;i<nConn;i++)
    connTab[i]->print();

};

//////////////////////////////////
// TrjFile member functions 
//////////////////////////////////

TrjFile::TrjFile(char* filename){

  int i;
  char* str1= new char[256];
  char* str2= new char[256];
  char* str3= new char[256];
  char* wellN = new char[256];
   
  int I,J,K; 
  double kx,ky,kz,ntg;
  double entry_md,entry_x,entry_y,entry_z;
  double exit_md,exit_x,exit_y,exit_z;
  

  antWells=0;
   
  
  pCurrent=new ifstream(filename);

/*
  filePath=new char[256];

  i =strlen(filename);
  printf("i is %i \n",i);

  while (filename[i]!='\\')
    i--;

  printf("i is %i \n",i);
  
  strcpy(filePath,filename);
  filePath[i+1]='\0';
  printf("File path: %s \n",filePath);
*/
   
  pCurrent->getline(str1,256);
  // printf(" %s \n\n",str1);	
 
 int l;
 
 
 while (!pCurrent->eof()){
  _seekKeyw_mf(this,"WELLNAME",str1);
  
   l=strlen(str1);
  
   if (str1[l-1]=='\r')
      str1[l-1]='\0';

   
  if (!pCurrent->eof()){  
  
    strcpy(wellN,str1); 
    textf::getStr(wellN,2);
    
    textf::replace2(wellN,'\'',' ');
    textf::trimL2(wellN);
    textf::trimR2(wellN);
    
        
    wellTab[antWells]=new Well(wellN);

            
    for (i=0;i<3;i++)
      pCurrent->getline(str1,256);
   
    i=0;
    strcpy(str2,"");
    
    while (strcmp(str2,"END_TRAJECTORY")!=0){
    
      pCurrent->getline(str1,256);

 
      l=strlen(str1);
  
      if (str1[l-1]=='\r')
        str1[l-1]='\0';

      strcpy(str2,str1);
      str2[14]='\0';

     // printf(">%s<  \n",str2);
      
      if (strcmp(str2,"END_TRAJECTORY")!=0){

        strcpy(str3,str1);
	textf::getStr2(str3,2);
        I=textf::toInt(str3);

        strcpy(str3,str1);
	textf::getStr2(str3,3);
        J=textf::toInt(str3);

        strcpy(str3,str1);
	textf::getStr2(str3,4);
        K=textf::toInt(str3);


        strcpy(str3,str1);
	textf::getStr2(str3,14);
        kx=textf::toDouble(str3);
 
        strcpy(str3,str1);
	textf::getStr2(str3,15);
        ky=textf::toDouble(str3);

        strcpy(str3,str1);
	textf::getStr2(str3,16);
        kz=textf::toDouble(str3);

        strcpy(str3,str1);
	textf::getStr2(str3,17);
        ntg=textf::toDouble(str3);
 
 
        strcpy(str3,str1);
	textf::getStr2(str3,1);
        entry_md=textf::toDouble(str3);
 
        strcpy(str3,str1);
	textf::getStr2(str3,5);
        entry_x=textf::toDouble(str3);
 
        strcpy(str3,str1);
	textf::getStr2(str3,6);
        entry_y=textf::toDouble(str3);
 
        strcpy(str3,str1);
	textf::getStr2(str3,7);
        entry_z=textf::toDouble(str3);
 
 
        strcpy(str3,str1);
	textf::getStr2(str3,9);
        exit_md=textf::toDouble(str3);
 
        strcpy(str3,str1);
	textf::getStr2(str3,10);
        exit_x=textf::toDouble(str3);
 
        strcpy(str3,str1);
	textf::getStr2(str3,11);
        exit_y=textf::toDouble(str3);
 
        strcpy(str3,str1);
	textf::getStr2(str3,12);
        exit_z=textf::toDouble(str3);


      
        wellTab[antWells]->addConnection(I,J,K,kx,ky,kz,ntg,entry_md,entry_x,entry_y,entry_z,exit_md,exit_x,exit_y,exit_z);
        i++;
      }      

    }
    
  //  wellTab[antWells]->printConn();  
    
    // printf("Broenn: %s  ant conn:  %i  \n",wellTab[antWells]->wellName,wellTab[antWells]->nConn);
    
    antWells++;


     
  }


 }

  // printf("\nAntall brønner: %i \n",antWells); 
  
  /*	
  if (pCurrent->eof())
    printf("Har naadd eof !! \n\n");
  else  
    printf("Har ikke nådd eof !! \n\n");

  pCurrent->getline(str1,256);
  printf(" %s \n\n",str1);	


 Well* well1=new Well("A-7");
*/

/*
  _seekKeyw_mf(this,"xxxxxxWELLNAME",str1);
  printf(" %s \n\n",str1);	

  if (pCurrent->eof())
    printf("Har naadd eof !! \n\n");
  else  
    printf("Har ikke nådd eof !! \n\n");
*/
  

};

TrjFile::~TrjFile(){

   delete pCurrent;
};

Well* TrjFile::findWell(char* wellN){

  Well* w1=NULL;
  int i;
  
  for (i=0;i<antWells;i++)
    if (strcmp(wellN,wellTab[i]->wellName)==0)
      w1=wellTab[i];
      
  return w1;
  
}


bool TrjFile::fileExist(char* filename){

  FILE *stream;
  bool res=true;

  if( (stream  = fopen( filename, "r" )) == NULL )
      res=false;
  
  return res;

}


TrjFile* TrjFile::seekKeyw(TrjFile* datafil,char* keyW){

  return TrjFile::_seekKeyw_mf(datafil,keyW);

}

TrjFile* TrjFile::_seekKeyw_mf(TrjFile* datafil,char* keyW){

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


TrjFile* TrjFile::_seekKeyw_mf(TrjFile* datafil,char* keyW,char* line){

    char* str=new char[256];
    char* test=new char[256];
    bool found=false;
    int l;
    int lkeyw=strlen(keyW);


    while ((!datafil->pCurrent->eof()) && (!found)){

	   datafil->pCurrent->getline(str,256);
/*
           l=strlen(line);
  
           if (line[l-1]=='\r')
              line[l-1]='\0';
*/	   
	   if (str[strlen(str)-1]=='\r')
             str[strlen(str)-1]='\0';

	   strcpy(line,str);

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



void TrjFile::nextLine(TrjFile* datafil,char* str){


    char* str1=new char[256];
    int pos=-1;
    int lstr=0;

    strcpy(str,"");
	  
    datafil->pCurrent->getline(str,256);
    textf::replace(str1,'\t',' ');

    lstr=strlen(str);
  
    if (str[lstr-1]=='\r')
      str[lstr-1]='\0';
    
}

void TrjFile::print(){

 int i,j;
 Well* w1;
 
 for (i=0;i<antWells;i++){
 
   printf("Broenn: %s \n",wellTab[i]->wellName);
   
   w1=wellTab[i];
   
   for (j=0;j<w1->nConn;j++){

     printf(" %3i %3i %3i \n",w1->connTab[j]->I,w1->connTab[j]->J,w1->connTab[j]->K);
     
   } 
   
   
 }
 
 
 
 

}

void TrjFile::print(char* fname){

 int i,j;
 Well* w1;
 char* str = new char[256];

 ofstream* utfil = new ofstream(fname);
 
 for (i=0;i<antWells;i++){
 
   sprintf(str,"Broenn: %s \n",wellTab[i]->wellName);
   *utfil << str;
   
   w1=wellTab[i];
   
   for (j=0;j<w1->nConn;j++){

     sprintf(str," %3i %3i %3i \n",w1->connTab[j]->I,w1->connTab[j]->J,w1->connTab[j]->K);
     *utfil << str;
     
   } 

 }

 utfil->flush();
 utfil->close();
   
 delete str;
 

}

