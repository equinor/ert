#include <iostream.h>
#include <stdio.h>
#include <fstream.h>

//#include "textf.h"

int const maxAntParam=100;

struct Node {

	double x,y,z;
		
};


class Cell {

private:
    double C(double *r,int i1,int i2,int i3);
    int perm123(int i1,int i2,int i3);

public:
    double distanceTo(Cell* c1);
    bool inside2(double x,double y);
    bool inside2(double x,double y,double z);
    bool inside1(double x,double y);
    bool inside3(double x,double y);
    bool inside3(double x,double y,double z);
    void print();
    void print(char* fName);

   Node* n;
   Cell();
   ~Cell();

   double top();
   double top(double x,double y);
   double base();   
   double base(double x,double y);   
   double calc_x();
   double calc_y();

   double calc_dx();
   double calc_dy();
   double calc_dz();

   double volume();

};



// forward declarations for class EclGrid !!!!
// can possibly be removed ???

class CWell;
class CPolygon;
class CProgDlg;

class EclGrid {


public:

    int nI,nJ,nK;
    int i1,i2,j1,j2; 
    Cell*** cell;
    int*** actnum;
	long nActiveCells;

	double minZ;
	double 	x_offset,y_offset;
	bool extended;
	int* antActive;
    char** layerTable;

	EclGrid();
	EclGrid(int _nI,int _nJ,int _nK);
	~EclGrid();

	void import_grid(char* name);

	void read_init(char* name);

	// void BuildConnections(CWell* well);
	
	void exportAsciiExcel(char* filename,int layer,double xOff,double yOff,int i1,int i2,int j1,int j2);


	double*** newInitParam(char* name);
	double*** getInitParam(char* name);
	int getNParam();



private:

    void newCells();
    long getInteger(ifstream* stream);
    float getFloat(ifstream* stream);
    long antByteFwdTo(char* str,ifstream* stream,long max);
	
   // init parameters

    double ****param;
    int nParam;
    char **paramName;


};

