/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'textf.h' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#include <stdlib.h>
#include <fstream.h>
//#include <afx.h>




class textf {


public:
	static void getFileRoot(char* str,char* root);

  static void getPath(char* path,char* str);
	
  static int toInt(char* _str);
  static double textf::toDouble(char* _str);

  static void nextStr(char* _str,ifstream* stream);
  static void nextStr(char* _str,ifstream* stream,char c);
  static void nextLine(char* _str,ifstream* stream);
  static int countStr(char* str);
  static double getDouble(char* str,int n);

  static char* getStr(char* str,int n);
  static void getStr2(char* str,int n);

  // static char* cstr2str(char* str,CString cstr);
  
  static char* trimL(char* str);
  static char* trimR(char* str);
  
  static void trimL2(char* str);
  static void trimR2(char* str);
  
  
  static char* toUpper(char* str);
  static void toUpper2(char* str);

  static char* replace(char* str,char r,char n);
  static void replace2(char* str,char r,char n);
  static char* inside(char* str,char c);
  static int inStr(char* str,char c,int pos);
  static int inStr(char* str1,char* str2);
  static int countNonSpace(char* str);
  static char* getItem(char* str,int n);
  static void getItem2(char* str,int n);
  static void getItem2(char* str,int n,int& p);
  static bool nullRecord(char* str);
  static char* addSpaceR(char* str,int n);
  static char* addStr(char* addTo,char* str);
  
  static void subStr(char* source,char* dest,int start,int ant);
  
  textf();
  ~textf();

  
};
