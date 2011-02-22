#  Copyright (C) 2011  Statoil ASA, Norway. 
#   
#  The file 'stringlist.py' is part of ERT - Ensemble based Reservoir Tool. 
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


import unittest
import libutil
import ctypes
from   ert.cwrap.cwrap import *



class StringList:
    def __init__( self , arg = None):
        self.c_ptr = cfunc.stringlist_alloc( )
        if arg:
            if isinstance( arg , types.ListType ) or isinstance( arg , types.TupleType ):
                for s in arg:
                    if isinstance( s , types.StringType):
                        self.append( s )
                    else:
                        raise TypeError("Item:%s not a string" % s)
                    
            else:
                raise TypeError("Stringlist( arg ) arg: must be list or tuple")

            
    def __del__( self ):
        cfunc.stringlist_free( self )


    def from_param( self ):
        return ctypes.c_void_p( self.c_ptr )


    def __getitem__(self , index):
        if isinstance( index , types.IntType):
            length = self.__len__()
            if index < 0 or index >= length:
                raise IndexError
            else:
                return cfunc.stringlist_iget( self , index )
        else:
            raise TypeError("Index should be integer type")


    def __len__(self):
        return cfunc.stringlist_get_size( self )


    def __str__(self):
        buffer = "["
        length = len(self)
        for i in range(length):
            if i == length -1:
                buffer += "\'%s\']" % self[i]
            else:
                buffer += "\'%s\'," % self[i]
        return buffer


    def append(self, s):
        if isinstance( s, types.StringType):
            cfunc.stringlist_append( self , s)
        else:
            sys.exit("Type mismatch")


    @property
    def strings(self):
        """
        Will return the strings in the stringlist as a 100% standard
        Python List of strings. 

        The content is copied, so the StringList() instance can very
        well go out of scope.
        """
        slist = []
        for s in self:
            slist.append( s )
        return slist



CWrapper.registerType( "stringlist" , StringList )

cwrapper = CWrapper( libutil.lib )
cfunc    = CWrapperNameSpace("StringList")
cfunc.stringlist_alloc      = cwrapper.prototype("c_void_p stringlist_alloc_new( )")
cfunc.stringlist_free       = cwrapper.prototype("void stringlist_free( stringlist )")
cfunc.stringlist_append     = cwrapper.prototype("void stringlist_append_copy( stringlist , char* )")
cfunc.stringlist_iget       = cwrapper.prototype("char* stringlist_iget( stringlist , int )")
cfunc.stringlist_get_size   = cwrapper.prototype("int stringlist_get_size( stringlist )") 

