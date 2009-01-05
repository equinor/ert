#include <iostream.h>
#include <stdio.h>
#include <fstream.h>
//#include "eclgrid.h"

const int maxAntWells = 999;

class EclGrid;

class Conn {

public:

   int I;
   int J;   
   int K;
   
   double kx;
   double ky;
   double kz;
   double ntg;
   
   double entry_md;
   double entry_x;
   double entry_y;
   double entry_z;
         
   double exit_md;
   double exit_x;
   double exit_y;
   double exit_z;
   double L;
   
   Conn();
   ~Conn();

   void print();
   double calcConn(EclGrid* simgrid,double S,double d,ofstream* debug);
   double calcConn(EclGrid* simgrid,double S,double d,ofstream* debug,double* _kh);
   void addLength(double _L);
   void setLength(double _L);
      
};


class Well {

public:

   int nConn;
   char* wellName; 
   Conn* connTab[999];

   void addConnection(int _I,int _J,int _K,double _kx,double _ky,double _kz,double _ntg,double _entry_md,double _entry_x,double _entry_y,double _entry_z,double _exit_md,double _exit_x,double _exit_y,double _exit_z);
   void printConn();
   
   Conn* findConnection(int _I,int _J,int _K);
   Conn* findConnection_test(int _I,int _J,int _K);
      
   Well();

   Well(char* _wellname);

   ~Well();

};


class TrjFile {

public:

   char* filePath; 
   int antWells;
   //Well* wellTab[maxAntWells];
    
   Well* wellTab[999]; 
   Well* findWell(char* wellN);
   
   ifstream* pCurrent;
   // EclFile* pParent;
    
   TrjFile(char* filename);

   void print();
   void print(char* fname);
    
   static bool fileExist(char* filename);
   static TrjFile* seekKeyw(TrjFile* datafil,char* keyW);

  // static void nextRecord(TrjFile* datafil,char* str,bool removeLineBreak);
  // static void nextRecord(TrjFile* datafil,char* str,bool removeLineBreak,bool removeComments);

   static void nextLine(TrjFile* datafil,char* str);

   ~TrjFile();

private:

   static TrjFile* _seekKeyw_mf(TrjFile* datafil,char* keyW);
   static TrjFile* _seekKeyw_mf(TrjFile* datafil,char* keyW,char* line);
   
   
  // static EclFile* _seekKeyw_withIncl(EclFile* datafil,char* keyW);

};
