/*
   Copyright (C) 2013  Statoil ASA, Norway. 
    
   The file 'plot_text.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#include <ert/util/test_util.h>

#include <ert/plot/plot_text.h>


bool test_create( ) {
  double xpos = 0.0;
  double ypos = 1.0;
  double font_scale = 0.07;
  
  const char * text = "Bjarne";
  bool OK = true;
  {
    plot_text_type * plot_text = plot_text_alloc( xpos , ypos , font_scale , text );
    plot_text_free( plot_text );
  }
  return OK;
}



int main(int argc , char ** argv) {
  
  test_assert_true( test_create() );
  
  exit(0);
}
