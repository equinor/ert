/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'plot.h' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#ifndef __plot_h__
#define __plot_h__
#ifdef __cplusplus
extern "c" {
#endif


#include <stdbool.h>

#include <ert/util/util.h>

#include <plplot/plplot.h>

#include <ert/plot/plot_const.h>
#include <ert/plot/plot_dataset.h>

stringlist_type * plot_refcase_fscanf(const char * plot_refcase_file );

#ifdef __cplusplus
}
#endif
#endif
