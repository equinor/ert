#  Copyright (C) 2011  Statoil ASA, Norway. 
#   
#  The file 'libutil.py' is part of ERT - Ensemble based Reservoir Tool. 
#   
#  ERT is free software: you can redistribute it and/or modify 
#  it under the terms of the GNU General Public License as published by 
#  the Free Software Foundation, either version 3 of the License, or 
#  (at your option) any later version. 
#   
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY 
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or 
#  FITNESS FOR A PARTICULAR PURPOSE.   
#   
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html> 
#  for more details. 


import ctypes
import ert.util.clib as clib
import ert.util.SDP  as SDP

clib.load("libz" , "libz.so.1")

# Between RedHat 3 and RedHat 4 they have changed which Fortran
# compiler has been used to compile blas / lapack - as a consequence
# additional libraries must be included on RedHat 3:
if SDP.RH_version() < 4:
    clib.load("libg2c.so.0")

clib.load("libblas.so" , "libblas.so.3")
clib.load("liblapack.so")
lib = clib.load("libutil.so")
