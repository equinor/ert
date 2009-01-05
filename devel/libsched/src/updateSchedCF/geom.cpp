#include "geom.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>    
#include <fstream.h>
#include <math.h>


//////////////////////////////////////////////////////
// Class vector2d member functions
//////////////////////////////////////////////////////


vector2d& vector2d::operator *(double value){
  
   vector2d* vect = new vector2d(0,0);

   vect->x = x*value;
   vect->y = y*value;

   return *vect;

}

vector2d& vector2d::operator +(vector2d v1){
  
   vector2d* vect = new vector2d(0,0);

   vect->x = x+v1.x;
   vect->y = y+v1.y;

   return *vect;

}

vector2d& vector2d::operator -(vector2d v1){
  
   vector2d* vect = new vector2d(0,0);

   vect->x = x-v1.x;
   vect->y = y-v1.y;

   return *vect;

}

vector2d& vector2d::normalize(){

  vector2d* vect = new vector2d(0,0);

  vect->x = x * (1.0/abs());
  vect->y = y * (1.0/abs());

  return *vect;

}

ostream& operator<<(ostream& os,vector2d& v){

	return os << "[" << setprecision(3) << setiosflags(ios::fixed) << v.x << 
		" , " << setprecision(3) << setiosflags(ios::fixed) << v.y <<

		" ] ";

}


vector2d::vector2d(double _x,double _y){
   x = _x;
   y = _y;
};

vector2d::~vector2d(){
};

double vector2d::abs(){
  
  return sqrt(powl(x,2)+powl(y,2));

}

double vector2d::dot(vector2d* v1,vector2d* v2){

   return (v1->x*v2->x + v1->y * v2->y);

}




void vector2d::print(){

	printf("[%0.2f,%0.2f] ",x,y);
}


//////////////////////////////////////////////////////
// Class vector3d member functions
//////////////////////////////////////////////////////


vector3d::vector3d(double _x,double _y,double _z){
   x = _x;
   y = _y;
   z = _z;
};

vector3d::~vector3d(){
};


void vector3d::print(){

	printf("[%0.4f,%0.4f,%0.4f] ",x,y,z);
}

vector3d& vector3d::cross(vector3d* v1,vector3d* v2){

  vector3d* vect = new vector3d(0,0,0);

  vect->x = v1->y*v2->z - v1->z*v2->y;
  vect->y = v1->z*v2->x - v1->x*v2->z;
  vect->z = v1->x*v2->y - v1->y*v2->x;

  return *vect;

}

double vector3d::abs(){
  
  return sqrt(powl(x,2)+powl(y,2)+powl(z,2));

}

vector3d& vector3d::normalize(){

  vector3d* vect = new vector3d(0,0,0);

  vect->x = x * (1.0/abs());
  vect->y = y * (1.0/abs());
  vect->z = z * (1.0/abs());

  return *vect;

}

vector3d& vector3d::operator +(vector3d v1){
  
   vector3d* vect = new vector3d(0,0,0);

   vect->x = x+v1.x;
   vect->y = y+v1.y;
   vect->z = z+v1.z;

   return *vect;

}

vector3d& vector3d::operator -(vector3d v1){
  
   vector3d* vect = new vector3d(0,0,0);

   vect->x = x-v1.x;
   vect->y = y-v1.y;
   vect->z = z-v1.z;

   return *vect;

}


vector3d& vector3d::GaussEli(double** A,vector3d* y){

  vector3d* vect = new vector3d(0,0,0);
  int i,j;
  double m;
  double* b=new double[3];
  
  b[0]=y->x;
  b[1]=y->y;
  b[2]=y->z;
   
/*
  for (i=0;i<3;i++)
    printf(" %10.5f %10.5f  %10.5f | %10.5f \n",A[0][i],A[1][i],A[2][i],b[i]);
  printf("\n\n"); 
*/

  for (i=1;i<3;i++){
  
    m=-A[0][i]/A[0][0];
  
    for (j=0;j<3;j++)
      A[j][i]=A[j][i]+m*A[j][0];    
    
    b[i]=b[i]+m*b[0];

  }

  m=-A[1][2]/A[1][1];
  for (j=1;j<3;j++)
    A[j][2]=A[j][2]+m*A[j][1];    

  b[2]=b[2]+m*b[1];

/*
  for (i=0;i<3;i++)
    printf(" %10.5f %10.5f  %10.5f | %10.5f \n",A[0][i],A[1][i],A[2][i],b[i]);
  printf("\n\n"); 
*/
  
  vect->z=b[2]/A[2][2];
  vect->y=(b[1]-A[2][1]*vect->z)/A[1][1];
  vect->x=(b[0]-A[2][0]*vect->z-A[1][0]*vect->y)/A[0][0];



  return *vect;

}




//////////////////////////////////////////////////////
// Class CPolygon member functions
//////////////////////////////////////////////////////




CPolygon::CPolygon(){
   
	name=new char[256];
    strcpy(name,"undefined");

	px = new double[maxPoints];
    py = new double[maxPoints];

	nPoints = 0;

	minX = 1.0E+49;
    minY = 1.0E+49;
}


CPolygon::CPolygon(char* _name){
   
	name=new char[256];
    strcpy(name,_name);

	px = new double[maxPoints];
    py = new double[maxPoints];

	nPoints = 0;

	minX = 1.0E+49;
    minY = 1.0E+49;
}

CPolygon::CPolygon(double x,double y,double rad,int nSect){

	int i;
    double pi = 3.1415926535;
   
	name=new char[256];
    strcpy(name,"undefined");

	px = new double[maxPoints];
    py = new double[maxPoints];

	nPoints = nSect+1;

	for (i=0;i<=nSect;i++){
	  px[i]=x+ cos((double)i/(double)nSect*2*pi)*rad;
	  py[i]=y+ sin((double)i/(double)nSect*2*pi)*rad;
	}

	minX =-rad-2;
	minY =-rad-2;

}

/*
void CPolygon::readAscii(char* filename){

    ifstream* in;
	char* str1 = new char[256];
    in=new ifstream(filename,ios::nocreate);

	while (!in->eof()){

        in->getline(str1,256);
		//textf::nextLine(str1,in);
        
		if (textf::countStr(str1)>1){
		  px[nPoints]=textf::getDouble(str1,1);;
          py[nPoints]=textf::getDouble(str1,2);;
			
		  if (nPoints > maxPoints){
		  	printf("\n\nIncrease variable maxPoints and recompile \n\n");
			exit(1);
		  }

		  if (px[nPoints] < minX)
		  	minX = px[nPoints];

		  if (py[nPoints] < minY)
			minY = py[nPoints];

		  nPoints++;

		}
			
	};

	minX -= 2;
	minY -= 2;

}

*/

void CPolygon::print()
{

	for (int i=0;i<nPoints;i++)
		printf("%10.2f %10.2f \n",px[i],py[i]);

	printf("\n\nMinX: %10.2f MinY %10.2f \n\n",minX,minY);
	
}

bool CPolygon::intersect(int p1,int p2, double x,double y){

	double **A,*b;
	double a,xi,yi;

	A    =new double*[2];
	A[0] =new double[2];
	A[1] =new double[2];
    b    =new double[2]; 

	A[0][0] = (y-yOut)/(x - xOut);
	A[0][1] = -1.0;
	b[0] = (y-yOut)/(x - xOut)*xOut-yOut;

	if (px[p1]==px[p2]){

		yi = (b[0]-A[0][0]*px[p1])/A[0][1];

	    if (py[p1] > py[p2]){
	 	  int tmp = p1;
		  p1 = p2;
		  p2 = tmp;
		}

		if ((yi > py[p1]) && (yi < py[p2]) && (yi > yOut) && (yi < y)) 
			return true;
		else
			return false;


	}

	if (py[p1]==py[p2]){
		
		xi = (b[0]-A[0][1]*py[p1])/A[0][0];

	    if (px[p1] > px[p2]){
	 	  int tmp = p1;
		  p1 = p2;
		  p2 = tmp;
		}

		if ((xi > px[p1]) && (xi < px[p2]) && (xi > xOut) && (xi < x)) 
			return true;
		else
			return false;
	}

	if ((px[p1]!=px[p2]) && (py[p1]!=py[p2])){
	   A[1][0] = (py[p2]-py[p1])/(px[p2]-px[p1]);
	   A[1][1] = -1.0;

	   b[1] = (py[p2]-py[p1])/(px[p2] - px[p1])*px[p1]-py[p1];

	   //printf("%10.3f %10.3f |  %10.3f \n",A[0][0],A[0][1],b[0]);
	   //printf("%10.3f %10.3f |  %10.3f \n\n",A[1][0],A[1][1],b[1]);

	   a = A[1][0]/A[0][0];

	   A[1][0]-=A[0][0]*a;
	   A[1][1]-=A[0][1]*a;
	   b[1]   -=b[0]*a;

	   //printf("%10.3f %10.3f |  %10.3f \n",A[0][0],A[0][1],b[0]);
	   //printf("%10.3f %10.3f |  %10.3f \n",A[1][0],A[1][1],b[1]);

	   if (A[1][1]!=0){
	      yi = b[1]/A[1][1];
	      xi = (b[0] - A[0][1]*yi)/A[0][0];
	   } else {
		   yi = yOut;
		   xi=xOut;
	   }


	   //printf("\nxi = %10.3f   yi = %10.3f \n\n",xi,yi);

	   if (px[p1] > px[p2]){
	 	 int tmp = p1;
		  p1 = p2;
		  p2 = tmp;
	   }


	   if ((xi >= xOut) && ( xi <= x) && (xi >= px[p1]) && (xi <= px[p2]))
		  return true;
	   else
		  return false;

	}


	return true;
}

bool CPolygon::inside(double x, double y)
{

	int nIntersect=0;


	if (x!=minX)
		xOut = minX;
	else
		xOut = minX-1.0;

	if (y!=minY)
		yOut = minY;
	else
		yOut = minY-1.0;
	

	for (int i=1;i<nPoints;i++)
		if ((px[i-1]!=px[i]) || (py[i-1]!=py[i]))
			if (intersect(i-1,i,x,y))
			   nIntersect++;

//			   printf(" % i til %i  -> intersecting !!! \n",i-1,i);
//			} else 
//			   printf(" % i til %i  -> not intersecting !!! \n",i-1,i);

	if ((px[nPoints-1]!=px[0]) || (py[nPoints-1]!=py[0]))
		if (intersect(nPoints-1,0,x,y))
		   nIntersect++;

		//		   printf(" % i til %i  -> intersecting !!! \n",nPoints-1,0);
//			
//		} else 
//			   printf(" % i til %i  -> not intersecting !!! \n",nPoints-1,0);


	if ((nIntersect%2)==0)
		return false;
	else
		return true;

}

void CPolygon::print_to_file(char *filename)
{

	ofstream* utfil = new ofstream(filename);

	for (int i=0;i<nPoints;i++)
		*utfil << px[i] << "\t" << py[i] << "\n";

	utfil->flush();
	utfil->close();

}

void CPolygon::calcMinValues(){

	int i;

	minX=1E100;
	minY=1E100;

	for (i=0;i<nPoints;i++){

		if (px[i]<minX)
			minX=px[i];

		if (py[i]<minY)
			minY=py[i];
	
	}


}

