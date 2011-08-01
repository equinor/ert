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
"""
Simple wrapping of stringlist 'class' from C library.

The stringlist type from the libutil C library is a simple structure
consisting of a vector of \0 terminated char pointers - slightly
higher level than the (char ** string , int size) convention.

For a pure Python application you should just stick with a normal
Python list of string objects; but when interfacing with the C
libraries there are situations where you might need to instantiate a
stringlist instance. 

The StringList constructor can take an optional argument which should
be an iterable consisting of strings, and the strings property will
return a normal python list of string objects, used in this way you
hardly need to notice that the StringList class is at play.
"""

import libutil
import types
import ctypes
from   ert.cwrap.cwrap import *



class StringList:
    def __init__( self , initial = None):
        """
        Creates a new stringlist instance.
        
        Creates a new stringlist instance. The optional argument
        @initial should be an iterable of strings which will be the
        initial content of the StringList; the content will be copied
        from the initial list:

            S = StringList( initial = ["My" , "name" , "is", "John" , "Doe"] )

        If an element in the @initial argument is not a string the
        TypeError exception will be raised.
        """
        self.c_ptr = cfunc.stringlist_alloc( )
        if initial:
            for s in initial:
                if isinstance( s , types.StringType):
                    self.append( s )
                else:
                    raise TypeError("Item:%s not a string" % s)
                
            
    def __del__( self ):
        cfunc.stringlist_free( self )

    def from_param( self ):
        return ctypes.c_void_p( self.c_ptr )

    def __getitem__(self , index):
        """
        Implements [] read operator on the stringlist.
        
        The __getitem__ method supports negative, i.e. from the right,
        indexing; but not slices.
        """
        if isinstance( index , types.IntType):
            length = self.__len__()
            if index < 0:
                index += length
            if index < 0 or index >= length:
                raise IndexError
            else:
                return cfunc.stringlist_iget( self , index )
        else:
            raise TypeError("Index should be integer type")


    def __len__(self):
        """
        The length of the list - used to support builtin len().
        """
        return cfunc.stringlist_get_size( self )


    def __str__(self):
        """
        String representation of list; used when calling print."
        """
        buffer = "["
        length = len(self)
        for i in range(length):
            if i == length -1:
                buffer += "\'%s\']" % self[i]
            else:
                buffer += "\'%s\'," % self[i]
        return buffer


    def append(self, s):
        """
        Appends a new string @s to list.
        """
        if isinstance( s, types.StringType):
            cfunc.stringlist_append( self , s)
        else:
            sys.exit("Type mismatch")


    @property
    def strings(self):
        """
        The strings in as a normal Python list of strings.

        The content is copied, so the StringList() instance can safely go
        out of scope after the call has completed.
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

