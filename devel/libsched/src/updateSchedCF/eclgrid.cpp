#include <string.h>
#include <fstream.h>
#include <math.h>                   
#include "eclgrid.h"
#include "geom.h"

//#include <afx.h>
//#include <afxwin.h>


/////////////////////////////////////////////
//  Member functions for class Cell
/////////////////////////////////////////////


ostream& flyttall(ostream& os){

	return os << setprecision(0) << setiosflags(ios::fixed);

}


Cell::Cell(){

  // default constructor

	n = new Node[8];

}

Cell::~Cell(){

}

double Cell::top(){

  return (n[0].z+n[1].z+n[2].z+n[3].z)/4.0;

}

double Cell::top(double x,double y){

// plane: d = -ax-by-cz-d = 0

	double d;

    vector3d v1(n[0].x-n[1].x,n[0].y-n[1].y,n[0].z-n[1].z);
    vector3d v2(n[3].x-n[1].x,n[3].y-n[1].y,n[3].z-n[1].z);

	
	vector3d v3 = vector3d::cross(&v1,&v2);


	d = - v3.x*n[0].x - v3.y*n[0].y - v3.z*n[0].z;

	return -1.0*(v3.x*x+v3.y*y+d)/v3.z;

}

double Cell::base(double x,double y){

// plane: d = -ax-by-cz-d = 0

	double d;

    vector3d v1(n[4].x-n[5].x,n[4].y-n[5].y,n[4].z-n[5].z);
    vector3d v2(n[7].x-n[5].x,n[7].y-n[5].y,n[7].z-n[5].z);
	vector3d v3 = vector3d::cross(&v1,&v2);

	d = - v3.x*n[4].x - v3.y*n[4].y - v3.z*n[4].z;

	return -1.0*(v3.x*x+v3.y*y+d)/v3.z;

}

double Cell::base(){

  return (n[4].z+n[5].z+n[6].z+n[7].z)/4.0;

}

double Cell::calc_x(){

	 return (n[0].x + n[1].x + n[2].x + n[3].x + n[4].x + 
		 n[5].x  + n[6].x + n[7].x)/8.0;


}

double Cell::calc_y(){

	double res=0.0;
 
	for (int i=0;i<8;i++)
		res +=n[i].y;

	 return res/8.0;

}

double Cell::calc_dx(){
  
   vector2d v1(n[1].x-n[0].x,n[1].y-n[0].y);
   vector2d v2(n[3].x-n[2].x,n[3].y-n[2].y);
   vector2d v3(n[5].x-n[4].x,n[5].y-n[4].y);
   vector2d v4(n[7].x-n[6].x,n[7].y-n[6].y);

   return (v1.abs()+v2.abs()+v3.abs()+v4.abs())/4.0;
	  
};

double Cell::calc_dy(){
  
   vector2d v1(n[2].x-n[0].x,n[2].y-n[0].y);
   vector2d v2(n[3].x-n[1].x,n[3].y-n[1].y);
   vector2d v3(n[6].x-n[4].x,n[6].y-n[4].y);
   vector2d v4(n[7].x-n[5].x,n[7].y-n[5].y);

   return (v1.abs()+v2.abs()+v3.abs()+v4.abs())/4.0;
	  
};

double Cell::calc_dz(){

   return (n[4].z+n[5].z+n[6].z+n[7].z-(n[0].z+n[1].z+n[2].z+n[3].z))/4.0;
	  
};

void Cell::print()
{

	for (int i=0;i<8;i++)
		printf("%i.  %6.2f %6.2f %6.2f \n",i+1,n[i].x,n[i].y,n[i].z);

}


void Cell::print(char* fName)
{

	char* str =new char[256];

	ofstream* utfil = new ofstream(fName);

	for (int i=0;i<8;i++){
		sprintf(str,"%i.  %6.2f %6.2f %6.2f \n",i+1,n[i].x,n[i].y,n[i].z);
		*utfil << str;
	}

	utfil->flush();
	utfil->close();

	delete str;

}


bool Cell::inside2(double x, double y)
{

	bool x_inside=false;
	bool y_inside=false;

	double x1 = n[0].x + (n[2].x-n[0].x)/(n[2].y-n[0].y)*(y-n[0].y);
	double x2 = n[1].x + (n[3].x-n[1].x)/(n[3].y-n[1].y)*(y-n[1].y);

	double y1 = n[0].y + (n[1].y-n[0].y)/(n[1].x-n[0].x)*(x-n[0].x);
	double y2 = n[3].y + (n[3].y-n[2].y)/(n[3].x-n[2].x)*(x-n[2].x);

	if (((x > x1) && (x < x2)) || ((x < x1) && (x>x2)))
		x_inside=true;

	if (((y > y1) && (y < y2)) || ((y < y1) && (y>y2)))
		y_inside=true;



	if ((x_inside) && (y_inside))
		return true;
	else
		return false;

}

bool Cell::inside2(double x, double y,double z)
{

	bool x_inside=false;
	bool y_inside=false;


	double x1 = n[0].x + (n[2].x-n[0].x)/(n[2].y-n[0].y)*(y-n[0].y);
	double x2 = n[1].x + (n[3].x-n[1].x)/(n[3].y-n[1].y)*(y-n[1].y);

	double y1 = n[0].y + (n[1].y-n[0].y)/(n[1].x-n[0].x)*(x-n[0].x);
	double y2 = n[3].y + (n[3].y-n[2].y)/(n[3].x-n[2].x)*(x-n[2].x);

	if (((x > x1) && (x < x2)) || ((x < x1) && (x>x2)))
		x_inside=true;

	if (((y > y1) && (y < y2)) || ((y < y1) && (y>y2)))
		y_inside=true;



	if ((x_inside) && (y_inside) && (z>top()) && (z < base())) 
		return true;
	else
		return false;

}


bool Cell::inside1(double x, double y)
{

		// calculates vector ei1

	vector2d n1n2(n[1].x-n[0].x,n[1].y-n[0].y);
    vector2d e1 = n1n2.normalize();

	vector2d n1x(x-n[0].x,y-n[0].y);
	vector2d e2 = n1x.normalize();

	vector2d n1a(0,0);
	n1a = e1*(vector2d::dot(&e1,&e2)*n1x.abs());

	vector2d ax = n1x - n1a;
	vector2d ei1 = ax.normalize();

	// calculates vector ei2

	vector2d n3n4(n[3].x-n[2].x,n[3].y-n[2].y);
    e1 = n3n4.normalize();


	vector2d n3x(x-n[2].x,y-n[2].y);
	
	e2 = n3x.normalize();

	vector2d n3a(0,0);
	n3a = e1*(vector2d::dot(&e1,&e2)*n3x.abs());

	ax = n3x - n3a;
	vector2d ei2 = ax.normalize();

	// calculates vector ej1

	vector2d n1n3(n[2].x-n[0].x,n[2].y-n[0].y);
    e1 = n1n3.normalize();
	e2 = n1x.normalize();

	n1a = e1*(vector2d::dot(&e1,&e2)*n1x.abs());

	ax = n1x - n1a;
	vector2d ej1 = ax.normalize();

	// calculates vector ej2

	vector2d n2n4(n[3].x-n[1].x,n[3].y-n[1].y);
    e1 = n2n4.normalize();

	vector2d n2x(x-n[1].x,y-n[1].y);
	e2 = n2x.normalize();

	vector2d n2a = e1*(vector2d::dot(&e1,&e2)*n2x.abs());

	ax = n2x - n2a;
	vector2d ej2 = ax.normalize();


	if ((vector2d::dot(&ei1,&ei2) < 0) &&
		 (vector2d::dot(&ej1,&ej2) < 0))
		 return true;
	else
		return false;

}

bool Cell::inside3(double x, double y)
{

	CPolygon pol1;

	pol1.nPoints=5;

	pol1.px[0]=n[0].x;
	pol1.px[1]=n[1].x;
	pol1.px[2]=n[3].x;
	pol1.px[3]=n[4].x;
	pol1.px[4]=n[0].x;
	
	pol1.py[0]=n[0].y;
	pol1.py[1]=n[1].y;
	pol1.py[2]=n[3].y;
	pol1.py[3]=n[4].y;
	pol1.py[4]=n[0].y;
	
    pol1.calcMinValues();
	

	return pol1.inside(x,y);
	
}

bool Cell::inside3(double x, double y,double z)
{

	CPolygon pol1;
    // bool tmp;

	pol1.nPoints=5;

	pol1.px[0]=n[0].x;
	pol1.px[1]=n[1].x;
	pol1.px[2]=n[3].x;
	pol1.px[3]=n[4].x;
	pol1.px[4]=n[0].x;
	
	pol1.py[0]=n[0].y;
	pol1.py[1]=n[1].y;
	pol1.py[2]=n[3].y;
	pol1.py[3]=n[4].y;
	pol1.py[4]=n[0].y;
	
    pol1.calcMinValues();
	
	if ((top()<=z) && (base()>=z) && (pol1.inside(x,y))) 
	  return true;
	else
	  return false;
	
}



int Cell::perm123(int i1,int i2,int i3){
 int temp;

  if ((i1==1) && (i2==2) && (i3==3))
	temp=0;

  if ((i1==1) && (i2==3) && (i3==2))
	temp=1;

  if ((i1==2) && (i2==1) && (i3==3))
	temp=1;

  if ((i1==2) && (i2==3) && (i3==1))
	temp=2;

  if ((i1==3) && (i2==1) && (i3==2))
	temp=2;

  if ((i1==3) && (i2==2) && (i3==1))
	temp=3;

  return temp;


}

double Cell::C(double *r,int i1,int i2,int i3){

 double temp;

 if ((i1==0) && (i2==0) && (i3==0))
	temp=r[0];

 if ((i1==1) && (i2==0) && (i3==0))
	temp=r[1]-r[0];

 if ((i1==0) && (i2==1) && (i3==0))
	temp=r[2]-r[0];

 if ((i1==0) && (i2==0) && (i3==1))
	temp=r[4]-r[0];

 if ((i1==1) && (i2==1) && (i3==0))
	temp=r[3]+r[0]-r[2]-r[1];

 if ((i1==0) && (i2==1) && (i3==1))
	temp=r[6]+r[0]-r[4]-r[2];

 if ((i1==1) && (i2==0) && (i3==1))
	temp=r[5]+r[0]-r[4]-r[1];

 if ((i1==1) && (i2==1) && (i3==1))
	temp=r[7]+r[4]+r[2]+r[1]-r[6]-r[5]-r[3]-r[0];


 return temp;

}

double Cell::volume(){

  double volume,X[8],Y[8],Z[8];
  int pb,pg,qa,qg,ra,rb;

  for (int i=0;i<=7;i++){
	 X[i]= this->n[i].x; 
	 Y[i]= this->n[i].y;
	 Z[i]= this->n[i].z;
  }

  volume=0.0;

 /* permutasjon (1,2,3)   */

  for (pb=0;pb<=1;pb++)
	for (pg=0;pg<=1;pg++)
	 for (qa=0;qa<=1;qa++)
	  for (qg=0;qg<=1;qg++)
		for (ra=0;ra<=1;ra++)
		 for (rb=0;rb<=1;rb++)
		  volume=volume+(pow(-1.0,perm123(1,2,3))*(C(X,1,pb,pg)*C(Y,qa,1,qg)*C(Z,ra,rb,1))/((qa+ra+1)*(pb+rb+1)*(pg+qg+1)));

  /* printf("%10.4f \n",volume); */

 /* permutasjon (1,3,2)   */

  for (pb=0;pb<=1;pb++)
	for (pg=0;pg<=1;pg++)
	 for (qa=0;qa<=1;qa++)
	  for (qg=0;qg<=1;qg++)
		for (ra=0;ra<=1;ra++)
		 for (rb=0;rb<=1;rb++)
			volume=volume+(pow(-1.0,perm123(1,3,2))*(C(X,1,pb,pg)*C(Z,qa,1,qg)*C(Y,ra,rb,1))/((qa+ra+1)*(pb+rb+1)*(pg+qg+1)));

 /* printf("%10.4f \n",volume); */


 /* permutasjon (2,1,3)   */

  for (pb=0;pb<=1;pb++)
	for (pg=0;pg<=1;pg++)
	 for (qa=0;qa<=1;qa++)
	  for (qg=0;qg<=1;qg++)
		for (ra=0;ra<=1;ra++)
		 for (rb=0;rb<=1;rb++)
			volume=volume+pow(-1.0,perm123(2,1,3))*(C(Y,1,pb,pg)*C(X,qa,1,qg)*C(Z,ra,rb,1))/((qa+ra+1)*(pb+rb+1)*(pg+qg+1));

  /* printf("%10.4f \n",volume); */


 /* permutasjon (2,3,1)   */

  for (pb=0;pb<=1;pb++)
	for (pg=0;pg<=1;pg++)
	 for (qa=0;qa<=1;qa++)
	  for (qg=0;qg<=1;qg++)
		for (ra=0;ra<=1;ra++)
		 for (rb=0;rb<=1;rb++)
			volume=volume+pow(-1.0,perm123(2,3,1))*(C(Y,1,pb,pg)*C(Z,qa,1,qg)*C(X,ra,rb,1))/((qa+ra+1)*(pb+rb+1)*(pg+qg+1));

  /* printf("%10.4f \n",volume); */

 /* permutasjon (3,1,2)   */

  for (pb=0;pb<=1;pb++)
	for (pg=0;pg<=1;pg++)
	 for (qa=0;qa<=1;qa++)
	  for (qg=0;qg<=1;qg++)
		for (ra=0;ra<=1;ra++)
		 for (rb=0;rb<=1;rb++)
			volume=volume+pow(-1.0,perm123(3,1,2))*(C(Z,1,pb,pg)*C(X,qa,1,qg)*C(Y,ra,rb,1))/((qa+ra+1)*(pb+rb+1)*(pg+qg+1));

  /* printf("%10.4f \n",volume); */

 /* permutasjon (3,2,1)   */

  for (pb=0;pb<=1;pb++)
	for (pg=0;pg<=1;pg++)
	 for (qa=0;qa<=1;qa++)
	  for (qg=0;qg<=1;qg++)
		for (ra=0;ra<=1;ra++)
		 for (rb=0;rb<=1;rb++)
			volume=volume+pow(-1.0,perm123(3,2,1)) *(C(Z,1,pb,pg)*C(Y,qa,1,qg)*C(X,ra,rb,1))/((qa+ra+1)*(pb+rb+1)*(pg+qg+1));

  /* printf("%10.4f \n",volume); */


  return fabs(volume);

}


double Cell::distanceTo(Cell *c1)
{

	return (sqrt(powl(c1->calc_x()-calc_x(),2)+
		        powl(c1->calc_y()-calc_y(),2)));
}



/////////////////////////////////////////////
//  Member functions for class EclGrid
/////////////////////////////////////////////




EclGrid::EclGrid(int _nI,int _nJ,int _nK){

  int i;
   
  nI=_nI;
  nJ=_nJ;
  nK=_nK;

  cell = new Cell **[nI];

  for (i=0;i<nI;i++)
	  cell[i]=new Cell *[nJ];

  for (i=0;i<nI;i++)
	  for (int j=0;j<nJ;j++)
		  cell[i][j] = new Cell[nK];


  actnum = new int**[nI];
  
  for (i=0;i<nI;i++)
	actnum[i]=new int*[nJ];

  for (i=0;i<nI;i++)
	  for (int j=0;j<nJ;j++)
		  actnum[i][j] = new int[nK];
	
  nParam=0;
  param=new double***[maxAntParam];
  paramName=new char*[maxAntParam];

  layerTable = new char*[nK];
  
  for (int k=0;k<nK;k++){
	layerTable[k]=new char[256];
    sprintf(layerTable[k],"layer%i",k+1);
  }



};

EclGrid::EclGrid(){


  nI=0;
  nJ=0;
  nK=0;

  cell = NULL;
  actnum=NULL;
  layerTable=NULL;

  nParam=0;
  param=new double***[maxAntParam];
  paramName=new char*[maxAntParam];


};


void EclGrid::newCells(){

  int i;
  cell = new Cell **[nI];

  for (i=0;i<nI;i++)
	  cell[i]=new Cell *[nJ];

  for (i=0;i<nI;i++)
	  for (int j=0;j<nJ;j++)
		  cell[i][j] = new Cell[nK];


  actnum = new int**[nI];
  
  for (i=0;i<nI;i++)
	actnum[i]=new int*[nJ];

  for (i=0;i<nI;i++)
	  for (int j=0;j<nJ;j++)
		  actnum[i][j] = new int[nK];

  layerTable = new char*[nK];
  
  for (int k=0;k<nK;k++){
	layerTable[k]=new char[256];
    sprintf(layerTable[k],"layer%i",k+1);
  }


}

EclGrid::~EclGrid(){

};





void EclGrid::import_grid(char* name){


   long ant;
   long forrigeAnt;
   char* message = new char[260];
   char* keyW = new char[10];
   char* keyWType = new char[10];
   int i,_i,_j,_k;   
   long ind,antCoords;
   double xf,yf,x0,y0,cf;

   int step=0;
   int prev=0;
   long pos=0;

   char* forrigeKeyW = new char[10];
   char* forrigeKeyWType = new char[10];

   long test1;
   long test2;
   long test3;
   long test;

   nActiveCells=0;

   test1=0;
   test2=0;
   test3=0;


   x0=0;    y0=0;
   xf=1.0;  yf=1.0;


   		//   xf=(x3-x0)/sqrt(powl((x3-x0),2)+powl((y3-y0),2));
		//   yf=(y1-y0)/sqrt(powl((x1-x0),2)+powl((y1-y0),2));

		//   cell[_i-1][_j-1][_k-1].n[i].x =x0+xf*getFloat(stream);
        //   cell[_i-1][_j-1][_k-1].n[i].y =y0+yf*getFloat(stream);


   ifstream* stream;

   if (cell!=NULL) {
	 delete cell;
   }


// 659: `nocreate' is not a member of type `std::basic_ios<char, std::char_traits<char> >'
//   stream=new ifstream(name,ios::binary | ios::nocreate);

   stream=new ifstream(name,ios::binary);
   
   if (!stream->good()){
     printf("\nCould not read gridfile %s \n\n",name);
     exit(1);
   }

  
   stream->seekg(0,ios::end);
   pos=stream->tellg();

   stream->seekg(0,ios::beg);
   pos=0;

   stream->seekg(4,ios::cur);
   
   antCoords=0;

   while (! stream->eof()){
     
     keyW[0]='_';

     stream->read(keyW,8);
     keyW[8]='\0';

     ant=getInteger(stream);

     stream->read(keyWType,4);   
     keyWType[4]='\0';

    //  printf("Reading %s  ant:  %i Type: %s \n",keyW,ant,keyWType);

     stream->seekg(8,ios::cur);
	 
     if (strcmp(keyWType,"INTE")==0){
     
        if (strcmp(keyW,"DIMENS  ")==0){

           nI=getInteger(stream);
           nJ=getInteger(stream);
           nK=getInteger(stream);

           //printf("Dimensjon:  %i %i %i \n",nI,nJ,nK); 
           antActive=new int[nK];
	   
	   for (int i=0;i<nK;i++)
	     antActive[i]=0;

	   newCells();

         }

         if (strcmp(keyW,"COORDS  ")==0){

           _i=getInteger(stream);
           _j=getInteger(stream);
           _k=getInteger(stream);

           antCoords++;

	   ind= (int)((_i+(_j-1)*nI+(_k-1)*nI*nJ)/1000);
 
           stream->seekg(4,ios::cur);
           actnum[_i-1][_j-1][_k-1]=getInteger(stream);

           antActive[_k-1]+=actnum[_i-1][_j-1][_k-1];

           nActiveCells+=actnum[_i-1][_j-1][_k-1];
           test1+=actnum[_i-1][_j-1][_k-1];

	   test=getInteger(stream);
		   
	   if (test>0)
             test2++;

           test=getInteger(stream);
		   
           if (test>0)
             test3++;


  	 }



         if ((strcmp(keyW,"COORDS  ")!=0) && (strcmp(keyW,"DIMENS  ")!=0)){
			
	    int shift=0;
	    if (ant > 1000) {
	      shift=ant / 1000;
	    }


	    stream->seekg(4*ant+shift*8,ios::cur);
	 }

     };  // if keyWType is 'INTE'

     if (strcmp(keyWType,"CHAR")==0){

       for (int i=0;i<ant;i++){
         stream->seekg(8,ios::cur);
       }

     };  // if keyWType is 'CHAR'

     if (strcmp(keyWType,"REAL")==0){


       if (strcmp(keyW,"CORNERS ")==0){
		   
	 for (i=0;i<8;i++){
			 
	   cell[_i-1][_j-1][_k-1].n[i].x =x0+xf*getFloat(stream);
           cell[_i-1][_j-1][_k-1].n[i].y =y0+yf*getFloat(stream);
           cell[_i-1][_j-1][_k-1].n[i].z =getFloat(stream);
		   
         }

      };
	   
       if (strcmp(keyW,"MAPAXES ")==0){
		   
         float x1,x3,y1,y3;

	 x1=getFloat(stream);
	 y1=getFloat(stream);
	 x0=getFloat(stream);
	 y0=getFloat(stream);
	 x3=getFloat(stream);
	 y3=getFloat(stream);
           
	 xf=(x3-x0)/sqrt(powl((x3-x0),2)+powl((y3-y0),2));
	 yf=(y1-y0)/sqrt(powl((x1-x0),2)+powl((y1-y0),2));



       };
		 
       if ((strcmp(keyW,"CORNERS ")!=0) && (strcmp(keyW,"MAPAXES ")!=0)){

         for (int i=0;i<ant;i++)
           stream->seekg(4,ios::cur);
       }

     };  // if keyWType is 'REAL'


     if ((strcmp(keyWType,"INTE")!=0) && (strcmp(keyWType,"CHAR")!=0) && (strcmp(keyWType,"REAL")!=0)) {
        
	sprintf(message,"Could not read keyword type %s \ni=%i,j=%i and k=%i",keyWType,_i,_j,_k);
	printf("%s\n\n",message);
	
	exit(1);

	 };

		
	 if (ant>0){
	   stream->seekg(8,ios::cur);
     }


     strcpy(forrigeKeyW,keyW);
     strcpy(forrigeKeyWType,keyWType);
     forrigeAnt=ant;


   } ;  // while not eof


   delete message;
   delete keyW;
   delete keyWType;
   delete stream;


   if (antCoords < (nI*nJ*nK))
      extended=false;
   else
      extended = true;


  // printf("Simulation grid succcessfully read ! \n\n");
   


};


void EclGrid::read_init(char* name){

  char* keyW = new char[10];
  char* keyWType = new char[10];

  char* forrigeKeyW = new char[10];
  char* forrigeKeyWType = new char[10];

  int forrigeAnt;

  char* message=new char[250];
  int ant;
  int i,j,k;
  char* str1 = new char[20];
  double cf;
  int n;
  double*** realArray;

  int pos;
  int prev=0;
  int step=0;
  int nPermx;

  ifstream* stream;


//914: `nocreate' is not a member of type `std::basic_ios<char, std::char_traits<char> >'
//  stream=new ifstream(name,ios::binary | ios::nocreate);

  stream=new ifstream(name,ios::binary);

  if (!stream->good()){
    printf("\nCould not read initfile %s \n\n",name);
    exit(1);
  }


  stream->seekg(0,ios::end);
  pos=stream->tellg();

  stream->seekg(0,ios::beg);
   
  pos=0;
  nPermx=0;

  stream->seekg(0,ios::end);
  stream->seekg(4,ios::beg);


  while (!stream->eof())  {

     keyW[0]='_';
     stream->read(keyW,8);
     keyW[8]='\0';

     ant=getInteger(stream);
 
     stream->read(keyWType,4);
     keyWType[4]='\0';

     stream->seekg(8,ios::cur);

     pos=stream->tellg();

     if (strcmp(keyWType,"INTE")==0){

       realArray=NULL;
 
       if (nActiveCells==ant){
 
         if ((strcmp(keyW,"EQLNUM  ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"PVTNUM  ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"SATNUM  ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"EOSNUM  ")==0))
            realArray=newInitParam(keyW);

        }

        if (realArray!=NULL){

          n=0;
	  for (k=0;k<nK;k++)
	   for (j=0;j<nJ;j++)
	     for (i=0;i<nI;i++){
	     	if (actnum[i][j][k]==1){	 
 	     	   realArray[i][j][k]=(double)getInteger(stream);
	     	   n++;	
	     	   if ((n%1000)==0)
	     	     stream->seekg(8,ios::cur);
	     	}
	     }
	     				 
	  stream->seekg(8,ios::cur);

	} else {

          stream->seekg(ant*4+(ant/1000)*8,ios::cur);
	  stream->seekg(8,ios::cur);

        }

     };  // if keyWType is 'INTE'

     if (strcmp(keyWType,"LOGI")==0){
		 
        stream->seekg(ant*4,ios::cur);
  	stream->seekg(8,ios::cur);

     };  // if keyWType is 'LOGI'

     if (strcmp(keyWType,"DOUB")==0){
		 
        stream->seekg(ant*8+(ant/1000)*8,ios::cur);
	stream->seekg(8,ios::cur);

     };  // if keyWType is 'LOGI'


     if (strcmp(keyWType,"MESS")==0){

       if (ant>0)
	      stream->seekg(16,ios::cur);

     };  // if keyWType is 'MESS'

     
	 
     if (strcmp(keyWType,"CHAR")==0){

        stream->seekg(ant*8,ios::cur);
  		   
	if (ant>0)
          stream->seekg(8,ios::cur);

     };  // if keyWType is 'CHAR'

     if (strcmp(keyWType,"REAL")==0){

       realArray=NULL;

       if (nActiveCells==ant){

	 if ((strcmp(keyW,"PERMX   ")==0))
           realArray=newInitParam(keyW);

	 if ((strcmp(keyW,"PERMY   ")==0))
           realArray=newInitParam(keyW);

	 if ((strcmp(keyW,"PERMZ   ")==0))
           realArray=newInitParam(keyW);

	 if ((strcmp(keyW,"NTG     ")==0))
           realArray=newInitParam(keyW);

	 if ((strcmp(keyW,"PORO    ")==0))
           realArray=newInitParam(keyW);
	 
	 if ((strcmp(keyW,"NTG     ")==0))
           realArray=newInitParam(keyW);

	 if ((strcmp(keyW,"MULTZ   ")==0))
           realArray=newInitParam(keyW);

       // props keywords type = 5

         if ((strcmp(keyW,"IKRG    ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"IKRGR   ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"IKRW    ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"IKRWR   ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"IKRO    ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"IKRORG  ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"IKRORW  ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"IPCG    ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"IPCW    ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"ISGL    ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"ISGCR   ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"ISGU    ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"ISWL    ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"ISWCR   ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"ISWU    ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"ISOGCR  ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"ISOWCR  ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"ISGLPC  ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"KRG     ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"KRGR    ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"KRO     ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"KRORG   ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"KRORW   ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"KRW     ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"KRWR    ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"PCG     ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"PCW     ")==0))
           realArray=newInitParam(keyW);
 
         if ((strcmp(keyW,"SGCR    ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"SGL     ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"SGLPC   ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"SGU     ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"SOGCR   ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"SOWCR   ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"SWATINIT")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"SWCR    ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"SWL     ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"SWLPC   ")==0))
           realArray=newInitParam(keyW);

         if ((strcmp(keyW,"SWU     ")==0))
           realArray=newInitParam(keyW);

       } // (nActiveCells==ant)



       if (realArray!=NULL){

          n=0;
	  for (k=0;k<nK;k++)
	   for (j=0;j<nJ;j++)
	     for (i=0;i<nI;i++){
		if (actnum[i][j][k]==1){	 
		   realArray[i][j][k]=getFloat(stream);
		   n++;	
		   if ((n%1000)==0)
		     stream->seekg(8,ios::cur);
		} 

             }
			 		   
	  stream->seekg(8,ios::cur);

       } else {  // (realArray!=NULL)

	   stream->seekg(ant*4+((ant/1000))*8,ios::cur);

           if (ant>0)
             stream->seekg(8,ios::cur);

	   } // (realArray!=NULL)
		

     };  // if keyWType is 'REAL'

     if ((strcmp(keyWType,"INTE")!=0) && (strcmp(keyWType,"CHAR")!=0) && (strcmp(keyWType,"REAL")!=0) && (strcmp(keyWType,"LOGI")!=0) && (strcmp(keyWType,"DOUB")!=0) && (strcmp(keyWType,"MESS")!=0)) {

	sprintf(message,"!! Could not read keyword type %s ",keyWType);
	printf("%s\n",message);
	
        exit(1); 
     };
 
     strcpy(forrigeKeyW,keyW);
     strcpy(forrigeKeyWType,keyWType);
     forrigeAnt=ant;
	 
  } ;  // while not eof

   
  delete str1;
  delete stream;


  delete keyW;
  delete keyWType;
  delete message;

  // printf("Simulation initfile succcessfully read ! \n\n");

};




long EclGrid::getInteger(ifstream* stream){

  unsigned char byte1;
  unsigned char byte2;
  unsigned char byte3;
  unsigned char byte4;
  int sign;
  long tmp;
  
  stream->read((char *)&byte4,sizeof(byte4));

  if (byte4>127) {
    sign=-1;
	byte4-=128;
  } else {
    sign=1;
  }

  stream->read((char *)&byte3,sizeof(byte3));
  stream->read((char *)&byte2,sizeof(byte2));
  stream->read((char *)&byte1,sizeof(byte1));

  tmp=sign*(byte1+byte2*256+byte3*256*256+byte4*256*256*256);


  return tmp;

};



float EclGrid::getFloat(ifstream* stream){

  unsigned char byte1;
  unsigned char byte2;
  unsigned char byte3;
  unsigned char byte4;
  int sign;
  double tmpFloat;

 // tmpFloat=44.44;
  
  long exp,mantisse;

  stream->read((char *)&byte4,sizeof(byte4));
  stream->read((char *)&byte3,sizeof(byte3));
  stream->read((char *)&byte2,sizeof(byte2));
  stream->read((char *)&byte1,sizeof(byte1));

  if (byte3>127){
	 mantisse=byte1+byte2*256+byte3*256*256;
  } else {
     mantisse = (long)powl(2,23)+byte1+byte2*256+byte3*256*256;
  }

  if (byte4 > 127){
	 byte4-=128;
	 sign=-1;
  } else {
     sign=1;
  }

  byte4*=2;

  if (byte3>127) {
    byte4+=1;
  }

  exp = byte4-127;

  tmpFloat=sign*((double)mantisse/powl(2,23-exp));


  return (float)tmpFloat;

};






long EclGrid::antByteFwdTo(char* str,ifstream* stream,long max){

  char* keyW=new char[8];
  char c;
  bool found;
  found=false;
  long ant;

  stream->read(keyW,8);
  keyW[8]='\0';
  ant=0;

  while ((!stream->eof()) && (strcmp(keyW,str)!=0) && (ant < max))  {
    
	stream->read(&c,1);
    
    for (int i=0;i<7;i++)
	  keyW[i]=keyW[i+1];

	keyW[7]=c;

	ant++;
  }

  return ant;

}






void EclGrid::exportAsciiExcel(char* filename,int layer,double xOff,double yOff,int i1,int i2,int j1, int j2){

	ofstream* utfil = new ofstream(filename);
    
	for (int j=j1;j<=j2;j++)
	  for (int i=i1;i<=i2;i++){

		if (actnum[i][j][layer]){
		  *utfil << cell[i][j][layer].n[0].x-xOff << "\t" << cell[i][j][layer].n[0].y-yOff << "\t" << cell[i][j][layer].n[0].z << "\n";
		  *utfil << cell[i][j][layer].n[1].x-xOff << "\t" << cell[i][j][layer].n[1].y-yOff << "\t" << cell[i][j][layer].n[1].z << "\n";
		  *utfil << cell[i][j][layer].n[3].x-xOff << "\t" << cell[i][j][layer].n[3].y-yOff << "\t" << cell[i][j][layer].n[3].z << "\n";
		  *utfil << cell[i][j][layer].n[2].x-xOff << "\t" << cell[i][j][layer].n[2].y-yOff << "\t" << cell[i][j][layer].n[2].z << "\n";
		  *utfil << cell[i][j][layer].n[0].x-xOff << "\t" << cell[i][j][layer].n[0].y-yOff << "\t" << cell[i][j][layer].n[0].z << "\n\n";

		  utfil->flush();
		}

	  }
   
	utfil->close();

}




double*** EclGrid::newInitParam(char* name){

  int i,j;
  
  if (nParam==maxAntParam){
	 char* message=new char[256];
     sprintf(message,"Error reading parameter <%s>. \nIncrease constant <maxAntParam> and recompile the program ffm2wlt",name);
     // AfxMessageBox(message,MB_ICONSTOP);
     
     // printf -> erstatt ned printf message
     
	 exit(1);
  }


  param[nParam]= new double**[nI];

  for (i=0;i<nI;i++)
    param[nParam][i]=new double*[nJ];
  
  for (i=0;i<nI;i++)
	for (j=0;j<nJ;j++)
	  param[nParam][i][j] = new double[nK];
	
  paramName[nParam]=new char[256];
  strcpy(paramName[nParam],name);

  nParam++;

  return param[nParam-1];

}

double*** EclGrid::getInitParam(char* name){

  int tmp;
  
  tmp=-1;

  for (int i=0;i<nParam;i++)
	if (strcmp(name,paramName[i])==0)
	  tmp=i;
    
  if (tmp==-1)
	  return NULL;
  else
	  return param[tmp];


}



int EclGrid::getNParam(){

	return nParam;
}

