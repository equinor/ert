/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'EclFile.h' is part of ERT - Ensemble based Reservoir Tool. 
    
   ERT is free software: you can redistribute it and/or modify 
   it under the terms of the GNU General Public License as published by 
   the Free Software Foundation, either version 3 of the License, or 
   (at your option) any later version. 
    
   ERT is distributed in the hope that it will be useful, but WITHOUT ANY 
   WARRANTY; without even the implied warranty of MERCHANTABILITY or 
   FITNESS FOR A PARTICULAR PURPOSE.   
    
   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html> 
   for more details. 
*/

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
