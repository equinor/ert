#  Copyright (C) 2011  Statoil ASA, Norway. 
#   
#  The file 'ecl_kw.py' is part of ERT - Ensemble based Reservoir Tool. 
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
Support for working with one keyword from ECLIPSE file.

ECLIPSE files in "restart format" are organized in keywords consisting
of a simple header and blocks of data. A keyword typically looks like:

  'SWAT    '  10000  'REAL'
  0.05  0.08  0.08  0.10
  0.11  0.11  0.10  0.09
  ....  

I.e. it starts with of header consisting of a 8 characters name, a
length and a datatype, immediately followed by the actual
data. 

Altough the term "restart format" is used to describe the format, this
particular format is not limited to restart files; it is (at least)
used in INIT, EGRID, GRID, Snnn, UNSMRY, SMSPEC, UNRST, Xnnnn and RFT
files. This module also has (some) support for working with GRDECL
'formatted' files.

The ecl_kw.py implementation wraps the ecl_kw.c implementation from
the libecl library.
"""

import  types
import  ctypes
import  numpy
from    ert.cwrap.cwrap       import *
from    ert.cwrap.cfile       import CFILE
from    ecl_util              import ECL_CHAR_TYPE, ECL_DOUBLE_TYPE, ECL_INT_TYPE, ECL_BOOL_TYPE, ECL_MESS_TYPE, ECL_FLOAT_TYPE 
import  ecl_util
import  fortio
import  libecl



        


class EclKW(object):
    """
    The EclKW class contains the information from one ECLIPSE keyword.

    The ecl_kw type is the lowest level type in the libecl C library,
    and all the other datatypes like e.g. ecl_grid and ecl_sum are
    based on collections of ecl_kw instances, and interpreting the
    content of the ecl_kw keywords.

    Many of the special __xxx___() functions have been implemented, so
    that the EclKW class supports both numerical operations and also
    [] based lookup. Many of the methods accept an optional @mask
    argument; this should be a EclRegion instance which can be used to
    limit the operation to a part of the EclKW.
    """

    @classmethod
    def new( cls , name, size , type):
        """
        Creates a brand new EclKW instance.

        This method will create a grand spanking new EclKW
        instance. The instance will get name @name (silently truncated
        to eight characters), @size elements and datatype @type. Using
        this method you could create a SOIL keyword with:

           soil_kw = EclKW.new( "SOIL" , 10000 , ECL_FLOAT_TYPE )
           
        """
        obj = cls()
        obj.c_ptr      = cfunc.alloc_new( name , size , type )
        obj.data_owner = True
        obj.parent     = None
        obj.__init( )
        return obj
    
    # Could this method be avoided totally by using an ecl_kw return
    # value from the ecl_file_iget_xxx() methods?
    @classmethod
    def ref( cls , c_ptr , parent ):
        obj = cls( )
        obj.c_ptr      = c_ptr
        obj.data_owner = False
        obj.parent     = parent
        obj.__init( )
        return obj


    @classmethod
    def slice_copy( cls , src , slice ):
        (start , stop , step) = slice.indices( src.size )
        if stop > start:
            obj = cls( )
            obj.c_ptr = cfunc.slice_copyc( src , start , stop , step)
            obj.parent = None
            obj.data_owner = True
            obj.__init( )
            return obj
        else:
            return EclKW.NULL( )
    

    @classmethod
    def copy( cls , src ):
        """
        Will create a deep copy of the current kw instance.
        """
        obj = cls( )
        obj.c_ptr = cfunc.copyc( src )
        obj.parent = None
        obj.data_owner = True
        obj.__init( )
        return obj
    
    @classmethod
    def NULL( cls ):
        """
        Utility method to create kw instance which evaluates to False.
        """
        obj = cls( )
        obj.c_ptr  = None
        obj.parent = None
        obj.data_owner = False
        return obj
    
    
    @classmethod
    def grdecl_load( cls , file , kw , ecl_type = ECL_FLOAT_TYPE):
        """
        Function to load an EclKW instance from a grdecl file.

        This constructor can be used to load an EclKW instance from a
        grdecl formatted file; the input files for petrophysical
        properties are typically given as grdecl files. 

        The @file argument should be a Python filehandle to an open
        file. The @kw argument should be the keyword header you are
        searching for, e.g. "PORO" or "PVTNUM"[1], the method will
        then search forward through the file to look for this @kw. If
        the keyword can not be found the method will return None.

        The grdecl files have no datatype header, so this must be
        supplied by the caller through the @ecl_type argument. The
        default value of @ecl_type is ECL_FLOAT_TYPE. 

           fileH =  open('ECLIPSE.PVT' , 'r')
           pvtnum_kw = EclKW.grdecl_load( fileH , 'PVTNUM', ecl_type = ECL_INT_TYPE)
           fileH.close()

           fileH = open('ECLIPSE.PETRO','r')
           poro_kw = EclKW.grdecl_load( fileH , 'PORO' )
           fileH.close()

        Observe that since the grdecl files are quite weakly
        structured it is difficult to verify the integrity of the
        files, malformed input might therefor pass unnoticed before
        things blow up at a later stage.
        
           
        [1]: It is possible, but not recommended, to pass in None for
             @kw, in which case the method will load the first keyword
             it finds in the file.
        """
        

        cfile  = CFILE( file )
        c_ptr  = cfunc.load_grdecl( cfile , kw , ecl_type )
        if c_ptr:
            obj = cls( )
            obj.c_ptr = c_ptr
            obj.data_owner = True
            obj.parent = None
            obj.__init()
            return obj
        else:
            return None

    def ecl_kw_instance( self ):
        return True


    def __init(self):
        self.data_ptr   = None
        self.ecl_type = cfunc.get_type( self )
        if self.ecl_type == ECL_INT_TYPE:
            self.data_ptr = cfunc.int_ptr( self )
            self.dtype    = numpy.int32        
            self.str_fmt  = "%8d "
        elif self.ecl_type == ECL_FLOAT_TYPE:
            self.data_ptr = cfunc.float_ptr( self )
            self.dtype    = numpy.float32
            self.str_fmt  = "%11.4f "
        elif self.ecl_type == ECL_DOUBLE_TYPE:
            self.data_ptr = cfunc.double_ptr( self )
            self.dtype    = numpy.float64        
            self.str_fmt  = "%11.4f "
        else:
            # Iteration not supported ...
            self.data_ptr = None
            self.dtype    = None
            self.str_fmt  = "%8s "


    def from_param(self):
        return self.c_ptr

    def __len__( self ):
        """
        Returns the number of elements. Implements len( )
        """
        return cfunc.get_size( self )


    def __nonzero__(self):
        if self.c_ptr:
            return True
        else:
            return False
    
    def __del__(self):
        if self.data_owner:
            cfunc.free( self )
            

    def __deep_copy__(self , memo):
        """
        Python special routine used to perform deep copy.
        """
        ecl_kw = EclKW.copy( self )
        return ecl_kw


    def __getitem__(self, index ):
        """
        Function to support index based lookup: y = kw[index]
        """
        if isinstance( index ,int ):
            length = self.__len__()
            if index < 0:
                # We allow one level of negative indexing
                index += self.size

            if index < 0 or index >= length:
                raise IndexError
            else:
                if self.data_ptr:
                    return self.data_ptr[ index ]
                else:
                    if self.ecl_type == ECL_BOOL_TYPE:
                        return cfunc.iget_bool( self, index)
                    elif self.ecl_type == ECL_CHAR_TYPE:
                        return cfunc.iget_char_ptr( self , index )
                    else:
                        raise TypeError("Internal implementation error ...")
        elif isinstance( index , slice):
            return self.slice_copy( self , index )
        else:
            raise TypeError("Index should be integer type")


    def __setitem__(self, index ,value):
        """
        Function to support index based assignment: kw[index] = value
        """
        if isinstance( index , types.IntType):
            length = self.__len__()
            if index < 0 or index >= length:
                raise IndexError
            else:
                if self.data_ptr:
                    self.data_ptr[ index ] = value
                else:
                    if self.ecl_type == ECL_BOOL_TYPE:
                        cfunc.iset_bool( self , index , value)
                    elif self.ecl_type == ECL_CHAR_TYPE:
                        return cfunc.iset_char_ptr( self , index , value)
                    else:
                        raise SystemError("Internal implementation error ...")
        else:
            raise TypeError("Index should be integer type")


    #################################################################
    

    def __IMUL__(self , factor , mul = True):
        if cfunc.assert_numeric( self ):
            if hasattr( factor , "ecl_kw_instance"):
                if cfunc.assert_binary( self, factor ):
                    if mul:
                        cfunc.imul( self , factor )
                    else:
                        cfunc.idiv( self , factor )
                else:
                    raise TypeError("Type mismatch")
            else:
                if not mul:
                    factor = 1.0 / factor
                    
                if self.ecl_type == ECL_INT_TYPE:
                    if isinstance( factor , int ):
                        cfunc.scale_int( self , factor )
                    else:
                        raise TypeError("Type mismatch")
                else:
                    if isinstance( factor , int ) or isinstance( factor , float):
                        cfunc.scale_float( self , factor )
                    else:
                        raise TypeError("Only muliplication with scalar supported")
        else:
            raise TypeError("Not numeric type")
        
        return self
                

    def __IADD__(self , delta , add = True):
        if cfunc.assert_numeric( self ):
            if type(self) == type(delta):
                if cfunc.assert_binary( self, delta):
                    if add:
                        cfunc.iadd(self , delta )
                    else:
                        cfunc.isub( self , delta )
                else:
                    raise TypeError("Type / size mismatch")
            else:
                if add:
                    sign = 1
                else:
                    sign = -1

                if self.ecl_type == ECL_INT_TYPE:
                    if isinstance( delta , int ):
                        cfunc.shift_int( self , delta * sign)
                    else:
                        raise TypeError("Type mismatch")
                else:
                    if isinstance( delta , int ) or isinstance( delta , float):
                        cfunc.shift_float( self , delta * sign ) # Will call the _float() or _double() function in the C layer.
                    else:
                        raise TypeError("Type mismatch")
        else:
            raise TypeError("Type / size mismatch")
        
        return self

    def __iadd__(self , delta):
        return self.__IADD__(delta , True )

    def __isub__(self , delta):
        return self.__IADD__(delta , False )

    def __imul__(self , delta):
        return self.__IMUL__(delta , True )

    def __idiv__(self , delta):
        return self.__IMUL__(delta , False )


    #################################################################
    
    
    def __add__(self , delta):
        copy = self.deep_copy()
        copy += delta
        return copy

    def __radd__(self, delta):
        return self.__add__( delta )

    def __sub__(self , delta):
        copy  = self.deep_copy()
        copy -= delta
        return copy

    def __rsub__( self , delta):
        return self.__sub__( delta ) * -1 
    
    def __mul__(self , factor):
        copy  = self.deep_copy()
        copy *= factor
        return copy

    def __rmul__(self , factor):
        return self.__mul__( factor )
    
    def __div__(self , factor):
        copy = self.deep_copy()
        copy /= factor
        return copy
    
    # No __rdiv__()

    def assert_binary( self , other ):
        """
        Utility function to assert that keywords @self and @other can
        be combined.
        """
        return cfunc.assert_binary( self , other )

    #################################################################
        
    def assign(self , value , mask = None , force_active = False):
        """
        Assign a value to current kw instance.

        This method is used to assign value(s) to the current EclKW
        instance. The @value parameter can either be a scalar, or
        another EclKW instance. To set all elements of a keyword to
        1.0:

            kw.assign( 1.0 )

        The numerical type of @value must be compatible with the
        current keyword. The optional @mask argument should be an
        EclRegion instance which can be used to limit the assignment
        to only parts of the EclKW. In the example below we select all
        the elements with PORO below 0.10, and then assign EQLNUM
        value 88 to those cells:
        
            grid = ecl.EclGrid("ECLIPSE.EGRID")
            reg  = ecl.EclRegion( grid , false )
            init = ecl.EclFile("ECLIPSE.INIT")

            poro = init.iget_named_kw( "PORO" , 0 )
            eqlnum = init.iget_named_kw( "EQLNUM" , 0 )
            reg.select_below( poro , 0.10 )
            
            eqlnum.assign( 88 , mask = reg )
        
        The EclRegion instance has two equivalent sets of selected
        indices; one consisting of active indices and one consisting
        of global indices. By default the assign() method will select
        the global indices if the keyword has nx*ny*nz elements and
        the active indices if the kw has nactive elements. By setting
        the optional argument @force_active to true, you can force the
        method to only modify the active indices, even though the
        keyword has nx*ny*nz elements; if the keyword has nactive
        elements the @force_active flag is not considered.
        """
        if cfunc.assert_numeric( self ):
            if type(value) == type(self):
                if mask:
                    mask.copy_kw( self , value , force_active)
                else:
                    if self.assert_binary( value ):
                        cfunc.copy_data( self , value)
                    else:
                        raise TypeError("Type / size mismatch")
            else:
                if mask:
                    mask.set_kw( self , value , force_active )
                else:
                    if self.ecl_type == ECL_INT_TYPE:
                        if isinstance( value , int ):
                            cfunc.set_int( self , value )
                        else:
                            raise TypeError("Type mismatch")
                    else:
                        if isinstance( value , int ) or isinstance( value, float):
                            cfunc.set_float( self , value )
                        else:
                            raise TypeError("Only muliplication with scalar supported")

                        
    def add( self , other , mask = None , force_active = False):
        """
        See method assign() for documentation of optional arguments
        @mask and @force_active.
        """
        if mask:
            mask.iadd_kw( self , other , force_active )
        else:
            return self.__iadd__( other )
        
    def sub(self , other , mask = None , force_active = False):
        """
        See method assign() for documentation of optional arguments
        @mask and @force_active.
        """
        if mask:
            mask.isub_kw(  self , other , force_active )
        else:
            return self.__isub__( other )

    def mul(self , other , mask = None , force_active = False):
        """
        See method assign() for documentation of optional arguments
        @mask and @force_active.
        """
        if mask:
            mask.imul_kw(  self , other , force_active )
        else:
            return self.__imul__( other )

    def div(self , other , mask = None , force_active = False):
        """
        See method assign() for documentation of optional arguments
        @mask and @force_active.
        """
        if mask:
            mask.idiv_kw(  self , other , force_active )
        else:
            return self.__idiv__( other )

    def apply( self , func , arg = None , mask = None , force_active = False):
        """
        Will apply the function @func on the keyword - inplace.

        The function @func should take a scalar value from the ecl_kw
        vector as input, and return a scalar value of the same type;
        optionally you can supply a second argument with the @arg
        attribute:

          def cutoff( x , limit):
              if x > limit:
                 return x
              else:
                 return 0


          kw.apply( math.sin )
          kw.apply( cutoff , arg = 0.10 )

        
        See method assign() for documentation of optional arguments
        @mask and @force_active.
        """
        if mask:
            active_list = mask.kw_index_list( self , force_active )
            if arg:
                for index in active_list:
                    self.data_ptr[index] = func( self.data_ptr[index] , arg)
            else:
                for index in active_list:
                    self.data_ptr[index] = func( self.data_ptr[index] )
        else:
            if arg:
                for i in range(self.size):
                    self.data_ptr[i] = func( self.data_ptr[i] , arg)
            else:
                for i in range(self.size):
                    self.data_ptr[i] = func( self.data_ptr[i] )
                    




    #################################################################

    def deep_copy( self ):
        ecl_kw = self.__deep_copy__( {} )
        return ecl_kw

    @property
    def size(self):
        return cfunc.get_size( self )
    
    def set_name( self , name ):
        cfunc.set_header( self , name )

    def get_name( self ):
        return cfunc.get_header( self )
        
    name = property( get_name , set_name )

    @property
    def type( self ):
        # enum ecl_type_enum from ecl_util.h
        if self.ecl_type == ECL_CHAR_TYPE:
            return "CHAR"
        if self.ecl_type == ECL_FLOAT_TYPE:
            return "REAL"
        if self.ecl_type == ECL_DOUBLE_TYPE:
            return "DOUB"
        if self.ecl_type == ECL_INT_TYPE:
            return "INTE"
        if self.ecl_type == ECL_BOOL_TYPE:
            return "BOOL"
        if self.ecl_type == ECL_MESS_TYPE:
            return "MESS"


    @property    
    def min_max( self ):
        if self.ecl_type == ECL_FLOAT_TYPE:
            min = ctypes.c_float()
            max = ctypes.c_float()
            cfunc.max_min_float( self , ctypes.byref( max ) , ctypes.byref( min ))
        elif self.ecl_type == ECL_DOUBLE_TYPE:
            min = ctypes.c_double()
            max = ctypes.c_double()
            cfunc.max_min_double( self , ctypes.byref( max ) , ctypes.byref( min ))
        elif self.ecl_type == ECL_INT_TYPE:
            min = ctypes.c_int()
            max = ctypes.c_int()
            cfunc.max_min_int( self , ctypes.byref( max ) , ctypes.byref( min ))
        else:
            print self.type
            return None
        return (min.value , max.value)


    @property
    def max( self ):
        min_max = self.min_max
        if min_max:
            return min_max[1]
        else:
            return None

    
    @property
    def min( self ):
        min_max = self.min_max
        if min_max:
            return min_max[0]
        else:
            return None
        
    
    @property
    def numeric(self):
        if self.ecl_type == ECL_FLOAT_TYPE:
            return True
        if self.ecl_type == ECL_DOUBLE_TYPE:
            return True
        if self.ecl_type == ECL_INT_TYPE:
            return True
        return False

    
    @property
    def type( self ):
        return self.ecl_type

    @property
    def type_name( self ):
        return ecl_util.type_name( self.ecl_type )
    
    @property
    def header( self ):
        return (self.name , self.size , self.type_name )


    def iget( self , index ):
        return self.__getitem__( index )
    
    
    @property
    def array(self):
        a = self.data_ptr
        if not a == None:
            a.size        = cfunc.get_size( self )
            a.__parent__  = self  # Inhibit GC
        return a

    
    def str_data( self , width , index1 , index2 , fmt):
        """
        Helper function for str() method.
        """
        data = []
        s = ""
        for index in range(index1, index2):
            data.append( self[index] )
        for index in range(len(data)):
            s += fmt % data[ index ]
            if index % width == (width - 1):
                s+= "\n"
        return s

        
    def str(self , width = 5 , max_lines = 10 , fmt = None):
        """
        Return string representation of kw for pretty printing.

        The function will return a string consisting of a header, and
        then a chunk of data. The data will be formatted in @width
        columns, and a maximum of @max_lines lines. If @max_lines is
        not sufficient the first elements in the kewyord are
        represented, a .... continuation line and then the last part
        of the keyword. If @max_lines is None all of the vector will
        be printed, irrespective of how long it is.

        If a value is given for @fmt that is used as format string for
        each element, otherwise a type-specific default format is
        used. If given the @fmt string should contain spacing between
        the elements. The implementation of the builtin method
        __str__() is based on this method.
        """
        s = "%-8s %8d %-4s\n" % (self.name , self.size , self.type_name)
        lines = self.size / width
        if not fmt:
            fmt = self.str_fmt

        if max_lines is None or lines <= max_lines:
            s += self.str_data( width , 0 , self.size , fmt)
        else:
            s1 = width * max_lines / 2
            s += self.str_data( width  , 0 , s1 , fmt)
            s += "   ....   \n"
            s += self.str_data( width  , self.size - s1 , self.size , fmt)
        
        return s

    def __str__(self):
        """
        Return string representation - see method str().
        """
        return self.str( width = 5 , max_lines = 10)
    


    @property
    def numpy_array( self ):
        if self.data_ptr:
            a = self.array
            value = numpy.zeros( a.size , dtype = self.dtype)
            for i in range( a.size ):
                value[i] = a[i]

    def fwrite( self , fortio ):
        cfunc.fwrite( self , fortio )

    def write_grdecl( self , file ):
        """
        Will write keyword in GRDECL format.

        This method will write the current keyword in GRDECL format,
        the @file argument must be a Python file handle to an already
        opened file. In the example below we load the porosity from an
        existing GRDECL file, set all poro values below 0.05 to 0.00
        and write back an updated GRDECL file.
        
            poro = ecl.EclKW.load_grdecl( open("poro1.grdecl" , "r") , "PORO" )
            grid = ecl.EclGrid( "ECLIPSE.EGRID" )
            reg  = ecl.EclRegion( grid , False )
            
            reg.select_below( poro , 0.05 )
            poro.assign( 0.0 , mask = reg )

            fileH = open( "poro2.grdecl" , "w")
            poro.write_grdecl( fileH )
            fileH.close()
            
        """
        cfile = CFILE( file ) 
        cfunc.fprintf_grdecl( self , cfile )




#################################################################

# 2. Creating a wrapper object around the libecl library, 
#    registering the type map : ecl_kw <-> EclKW
cwrapper = CWrapper( libecl.lib )
cwrapper.registerType( "ecl_kw" , EclKW )

# 3. Installing the c-functions used to manipulate ecl_kw instances.
#    These functions are used when implementing the EclKW class, not
#    used outside this scope.
cfunc = CWrapperNameSpace("ecl_kw")
cfunc.alloc_new                  = cwrapper.prototype("c_void_p ecl_kw_alloc( char* , int , int )")
cfunc.load_grdecl                = cwrapper.prototype("c_void_p ecl_kw_fscanf_alloc_grdecl_dynamic( FILE , char* , int )")
cfunc.copyc                      = cwrapper.prototype("c_void_p ecl_kw_alloc_copy( ecl_kw )")
cfunc.slice_copyc                = cwrapper.prototype("c_void_p ecl_kw_alloc_slice_copy( ecl_kw , int , int , int )")
cfunc.get_size                   = cwrapper.prototype("int      ecl_kw_get_size( ecl_kw )")
cfunc.get_type                   = cwrapper.prototype("int      ecl_kw_get_type( ecl_kw )")
cfunc.iget_char_ptr              = cwrapper.prototype("char*    ecl_kw_iget_char_ptr( ecl_kw , int )")
cfunc.iset_char_ptr              = cwrapper.prototype("void     ecl_kw_iset_char_ptr( ecl_kw , int , char*)")
cfunc.iget_bool                  = cwrapper.prototype("bool     ecl_kw_iget_bool( ecl_kw , int)")
cfunc.iset_bool                  = cwrapper.prototype("bool     ecl_kw_iset_bool( ecl_kw , int, bool)")
cfunc.iget_int                   = cwrapper.prototype("int      ecl_kw_iget_int( ecl_kw , int )")
cfunc.iget_double                = cwrapper.prototype("double   ecl_kw_iget_double( ecl_kw , int )")
cfunc.iget_float                 = cwrapper.prototype("float    ecl_kw_iget_float( ecl_kw , int)")
cfunc.float_ptr                  = cwrapper.prototype("float*   ecl_kw_get_float_ptr( ecl_kw )")
cfunc.int_ptr                    = cwrapper.prototype("int*     ecl_kw_get_int_ptr( ecl_kw )")
cfunc.double_ptr                 = cwrapper.prototype("double*  ecl_kw_get_double_ptr( ecl_kw )")
cfunc.free                       = cwrapper.prototype("void     ecl_kw_free( ecl_kw )")
cfunc.fwrite                     = cwrapper.prototype("void     ecl_kw_fwrite( ecl_kw , fortio )")
cfunc.get_header                 = cwrapper.prototype("char*    ecl_kw_get_header ( ecl_kw )")
cfunc.set_header                 = cwrapper.prototype("void     ecl_kw_set_header_name ( ecl_kw , char*)")
cfunc.fprintf_grdecl             = cwrapper.prototype("void     ecl_kw_fprintf_grdecl( ecl_kw , FILE )")
cfunc.fseek_grdecl               = cwrapper.prototype("bool     ecl_kw_grdecl_fseek_kw(char* , bool, bool , FILE )")

cfunc.iadd                       = cwrapper.prototype("void     ecl_kw_inplace_add( ecl_kw , ecl_kw )")
cfunc.imul                       = cwrapper.prototype("void     ecl_kw_inplace_mul( ecl_kw , ecl_kw )")
cfunc.idiv                       = cwrapper.prototype("void     ecl_kw_inplace_div( ecl_kw , ecl_kw )")
cfunc.isub                       = cwrapper.prototype("void     ecl_kw_inplace_sub( ecl_kw , ecl_kw )")

cfunc.assert_binary              = cwrapper.prototype("bool     ecl_kw_assert_binary_numeric( ecl_kw , ecl_kw )")
cfunc.scale_int                  = cwrapper.prototype("void     ecl_kw_scale_int( ecl_kw , int )")
cfunc.scale_float                = cwrapper.prototype("void     ecl_kw_scale_float_or_double( ecl_kw , double )")
cfunc.shift_int                  = cwrapper.prototype("void     ecl_kw_shift_int( ecl_kw , int )")
cfunc.shift_float                = cwrapper.prototype("void     ecl_kw_shift_float_or_double( ecl_kw , double )")
cfunc.assert_numeric             = cwrapper.prototype("bool     ecl_kw_assert_numeric( ecl_kw )")
cfunc.copy_data                  = cwrapper.prototype("void     ecl_kw_memcpy_data( ecl_kw , ecl_kw )")
cfunc.set_int                    = cwrapper.prototype("void     ecl_kw_scalar_set_int( ecl_kw , int )")
cfunc.set_float                  = cwrapper.prototype("void     ecl_kw_scalar_set_float_or_double( ecl_kw , double )")

cfunc.max_min_int                = cwrapper.prototype("void     ecl_kw_max_min_int( ecl_kw , int* , int*)")
cfunc.max_min_float              = cwrapper.prototype("void     ecl_kw_max_min_float( ecl_kw , float* , float*)")
cfunc.max_min_double             = cwrapper.prototype("void     ecl_kw_max_min_double( ecl_kw , double* , double*)")


