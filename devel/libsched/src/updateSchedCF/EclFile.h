#include <iostream.h>
#include <stdio.h>
#include <fstream.h>


class EclFile {

public:

   char* filePath; 

   ifstream* pCurrent;
   EclFile* pParent;
    
   EclFile(char* filename);
   EclFile(char* filename,EclFile* _pParent);

   static bool fileExist(char* filename);
   static EclFile* seekKeyw(EclFile* datafil,char* keyW,bool seekOnlyMainFile);

   static void nextRecord(EclFile* datafil,char* str,bool removeLineBreak);
   static void nextRecord(EclFile* datafil,char* str,bool removeLineBreak,bool removeComments);

   static void nextLine(EclFile* datafil,char* str);

   ~EclFile();

private:

   static EclFile* _seekKeyw_mf(EclFile* datafil,char* keyW);
   static EclFile* _seekKeyw_withIncl(EclFile* datafil,char* keyW);

};
