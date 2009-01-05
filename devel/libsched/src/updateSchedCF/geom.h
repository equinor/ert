#include <fstream.h>
#include <iomanip.h>
#include <math.h>


class vector2d {

public:
	double x,y;
    vector2d(double _x,double _y);     
    ~vector2d();     

	double abs();
    static double dot(vector2d* v1,vector2d* v2);

	vector2d& operator*(double);
	vector2d& operator+(vector2d);
	vector2d& operator-(vector2d);

	friend ostream& operator<< (ostream& os,vector2d& v);

	vector2d& normalize();
    void vector2d::print();
	
};



class vector3d {

public:

    double x,y,z;
    vector3d(double _x,double _y,double _z);     
    ~vector3d();     

    double abs();

    void print();
	
    static vector3d& cross(vector3d* v1,vector3d* v2);
    vector3d& normalize();
    static vector3d& GaussEli(double** A,vector3d* y);

    vector3d& operator+(vector3d);
    vector3d& operator-(vector3d);

};


const int maxPoints = 1000;


class CPolygon {

private:
	double xOut,yOut;
	bool intersect(int p1,int p2, double x,double y);
public:
    char* name;

	void print_to_file(char* filename);
	bool inside(double x,double y);
	void print();
	double minX,minY;
	int nPoints;
	double *px;
	double *py;

	CPolygon();
	CPolygon(char* name);
	CPolygon(double x,double y,double rad,int nSect);

	// void readAscii(char* filename);
        void calcMinValues();


};

