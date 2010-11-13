import ctypes
from   ert.cwrap.cwrap       import *
import ert.util.stringlist
import numpy
import fortio
import libecl
import ecl_kw
from   ert.util.cfile   import CFILE

class EclGrid(object):
    
    @classmethod
    def create(cls , specgrid , zcorn , coord , actnum , mapaxes = None ):
        obj = object.__new__( cls )
        gridhead = ecl_kw.EclKW.new( "GRIDHEAD" , 4 , 3 )
        gridhead[0] = 1
        for i in (range(3)):
            gridhead[i+1] = specgrid[i]

        obj.c_ptr = cfunc.grdecl_create( gridhead , zcorn , coord , actnum , None) # , mapaxes) 
        obj.data_owner = True
        obj.parent     = None
        return obj


    def __new__(cls , filename , lgr = None , parent = None):
        if filename:
            c_ptr = cfunc.fread_alloc( filename )
        elif lgr:
            c_ptr = lgr
            
        if c_ptr:
            obj = object.__new__( cls )
            obj.c_ptr = c_ptr
            if lgr:
                obj.data_owner = False
                obj.parent     = parent    # Keep a reference to the parent to inhibit GC.
            else:
                obj.data_owner = True
                obj.parent     = None
            return obj
        else:
            return None

        
    def __del__(self):
        if self.data_owner:
            cfunc.free( self )


    def from_param(self):
        return self.c_ptr

    @property
    def nx( self ):
        return cfunc.get_nx( self )

    @property
    def ny( self ):
        return cfunc.get_ny( self )

    @property
    def nz( self ):
        return cfunc.get_nz( self )

    @property
    def size( self ):
        return cfunc.get_nx( self ) * cfunc.get_ny( self ) * cfunc.get_nz( self )

    @property
    def nactive( self ):
        return cfunc.get_active( self )

    @property
    def dims( self ):
        return ( cfunc.get_nx( self ) ,
                 cfunc.get_ny( self ) ,
                 cfunc.get_nz( self ) ,
                 cfunc.get_active( self ) )

    @property
    def name( self ):
        return cfunc.get_name( self )

    def __global_index( self , active_index = None , global_index = None , ijk = None):
        set_count = 0
        if not active_index == None:
            set_count += 1
        if not global_index == None:
            set_count += 1
        if ijk:
            set_count += 1
            
        if not set_count == 1:
            raise ValueError("Exactly one of the kewyord arguments active_index, global_index or ijk must be set")
        
        if active_index:
            global_index = cfunc.get_global_index1A( self , active_index )
        elif ijk:
            global_index = cfunc.get_global_index3( self , ijk[0] , ijk[1] , ijk[2])
        
        return global_index
                 

    def get_active_index( self , ijk = None , global_index = None):
        gi = self.__global_index( global_index = global_index , ijk = ijk)
        return cfunc.get_active_index1( self , gi)


    def active( self , ijk = None , global_index = None):
        gi = self.__global_index( global_index = global_index , ijk = ijk)
        active_index = cfunc.get_active_index1( self , gi)
        if active_index >= 0:
            return True
        else:
            return False

    def get_global_index( self , ijk = None , active_index = None):
        gi = self.__global_index( active_index = active_index , ijk = ijk)
        return gi

    def get_ijk( self, active_index = None , global_index = None):
        i = ctypes.c_int()
        j = ctypes.c_int()
        k = ctypes.c_int()

        gi = self.__global_index( active_index = active_index , global_index = global_index)
        cfunc.get_ijk1( self , gi , ctypes.byref(i) , ctypes.byref(j) , ctypes.byref(k))

        return (i.value , j.value , k.value)


    def get_xyz( self, active_index = None , global_index = None , ijk = None):
        gi = self.__global_index( ijk = ijk , active_index = active_index , global_index = global_index)

        x = ctypes.c_double()
        y = ctypes.c_double()
        z = ctypes.c_double()

        cfunc.get_xyz1( self , gi , ctypes.byref(x) , ctypes.byref(y) , ctypes.byref(z))
        
        return (x.value , y.value , z.value)

    def depth( self , active_index = None , global_index = None , ijk = None):
        gi = self.__global_index( ijk = ijk , active_index = active_index , global_index = global_index)
        return cfunc.get_depth( self , gi )

    def find_cell( self , x , y , z , start_ijk = None):
        if start_ijk:
            start_index = self.__global_index( ijk = start_ijk )
        else:
            start_index = 0
        global_index = cfunc.get_ijk_xyz( self , x , y , z , start_index)
        if global_index >= 0:
            i = ctypes.c_int()
            j = ctypes.c_int()
            k = ctypes.c_int()
            cfunc.get_ijk1( self , global_index , ctypes.byref(i) , ctypes.byref(j) , ctypes.byref(k))        
            return (i.value , j.value , k.value)
        else:
            return None

    def cell_volume( self, active_index = None , global_index = None , ijk = None):
        gi = self.__global_index( ijk = ijk , active_index = active_index , global_index = global_index)
        return cfunc.get_cell_volume( self , gi)
            

    def cell_dz( self , active_index = None , global_index = None , ijk = None):
        gi = self.__global_index( ijk = ijk , active_index = active_index , global_index = global_index )
        return cfunc.get_cell_thickness( self , gi )


    @property
    def num_lgr( self ):
        return cfunc.num_lgr( self )


    def has_lgr( self , lgr_name ):
        if cfunc.has_lgr( self , lgr_name ):
            return True
        else:
            return False


    def get_lgr( self , lgr_name ):
        if cfunc.has_lgr(self , lgr_name ):
            lgr = EclGrid( None , lgr = cfunc.get_lgr( self , lgr_name ) , parent = self)
            return lgr
        else:
            return None
        

    def get_cell_lgr( self, active_index = None , global_index = None , ijk = None):
        gi  = self.__global_index( ijk = ijk , active_index = active_index , global_index = global_index)
        lgr = cfunc.get_cell_lgr( self , gi )
        if lgr:
            return EclGrid( None , lgr = lgr , parent = self)
        else:
            return None


    def grid_value( self , kw , i,j,k):
        return cfunc.grid_value( self , kw , i , j , k)


    def createKW( self , array , kw_name , pack):
        if array.ndim == 3:
            dims = array.shape
            if dims[0] == self.nx and dims[1] == self.ny and dims[2] == self.nz:
                dtype = array.dtype
                if dtype == numpy.int32:
                    type = ecl_kw.ECL_INT_TYPE
                elif dtype == numpy.float32:
                    type = ecl_kw.ECL_FLOAT_TYPE
                elif dtype == numpy.float64:
                    type = ecl_kw.ECL_DOUBLE_TYPE
                else:
                    sys.exit("Do not know how to create ecl_kw from type:%s" % dtype)
  
                if pack:
                    size = self.nactive
                else:
                    size = self.size
                    
                if len(kw_name) > 8:
                    # Silently truncate to length 8 - ECLIPSE has it's challenges.
                    kw_name = kw_name[0:8]  

                kw = ecl_kw.EclKW.new( kw_name , size , type )
                active_index = 0
                global_index = 0
                for k in range( self.nz ):
                    for j in range( self.ny ):
                        for i in range( self.nx ):
                            if pack:
                                if self.active( global_index = global_index ):
                                    kw[active_index] = array[i,j,k]
                                    active_index += 1
                            else:
                                if dtype == numpy.int32:
                                    kw[global_index] = int( array[i,j,k] )
                                else:
                                    kw[global_index] = array[i,j,k]
                                
                            global_index += 1
                return kw
        raise ValueError("Wrong size / dimension on array")

    
    

    def create3D( self , ecl_kw , default = 0):
        if ecl_kw.size == self.nactive or ecl_kw.size == self.size:
            array = numpy.ones( [ self.nx , self.ny , self.nz] , dtype = ecl_kw.dtype) * default
            array = numpy.ones( [ self.size ] , dtype = ecl_kw.dtype) * default
            kwa = ecl_kw.array
            if ecl_kw.size == self.size:
                for i in range(kwa.size):
                    array[i] = kwa[i]
            else:
                data_index = 0
                for global_index in range(self.size):
                    if self.active( global_index = global_index ):
                        array[global_index] = kwa[data_index]
                        data_index += 1
                        
            array = array.reshape( [self.nx , self.ny , self.nz] , order = 'F')
            return array
        else:
            raise ValueError("Keyword: %s has invalid size(%d), must be either nactive:%d  or nx*ny*nz:%d" % (ecl_kw.name , ecl_kw.size , self.nactive ,self.size))
        
        
    def write_grdecl( self , ecl_kw , pyfile , default_value = 0):
        if ecl_kw.size == self.nactive or ecl_kw.size == self.size:
            cfile = CFILE( pyfile )
            cfunc.fwrite_grdecl( self , ecl_kw , cfile , default_value )
        else:
            raise ValueError("Keyword: %s has invalid size(%d), must be either nactive:%d  or nx*ny*nz:%d" % (ecl_kw.name , ecl_kw.size , self.nactive ,self.size))



# 2. Creating a wrapper object around the libecl library, 
#    registering the type map : ecl_kw <-> EclKW
cwrapper = CWrapper( libecl.lib )
cwrapper.registerType( "ecl_grid" , EclGrid )

# 3. Installing the c-functions used to manipulate ecl_kw instances.
#    These functions are used when implementing the EclKW class, not
#    used outside this scope.
cfunc = CWrapperNameSpace("ecl_grid")


cfunc.fread_alloc                  = cwrapper.prototype("long ecl_grid_load_case( char* )")
cfunc.grdecl_create                = cwrapper.prototype("long ecl_grid_alloc_GRDECL_kw( ecl_kw , ecl_kw , ecl_kw , ecl_kw )") # MAPAXES not supported yet
cfunc.exists                       = cwrapper.prototype("bool ecl_grid_exists( char* )")
cfunc.free                         = cwrapper.prototype("void ecl_grid_free( ecl_grid )")     
cfunc.get_nx                       = cwrapper.prototype("int ecl_grid_get_nx( ecl_grid )")
cfunc.get_ny                       = cwrapper.prototype("int ecl_grid_get_ny( ecl_grid )")
cfunc.get_nz                       = cwrapper.prototype("int ecl_grid_get_nz( ecl_grid )")
cfunc.get_active                   = cwrapper.prototype("int ecl_grid_get_active_size( ecl_grid )")
cfunc.get_name                     = cwrapper.prototype("char* ecl_grid_get_name( ecl_grid )")
cfunc.get_active_index3            = cwrapper.prototype("int ecl_grid_get_active_index3( ecl_grid , int , int , int)")
cfunc.get_global_index3            = cwrapper.prototype("int ecl_grid_get_global_index3( ecl_grid , int , int , int)") 
cfunc.get_active_index1            = cwrapper.prototype("int ecl_grid_get_active_index1( ecl_grid , int )") 
cfunc.get_global_index1A           = cwrapper.prototype("int ecl_grid_get_global_index1A( ecl_grid , int )") 
cfunc.get_ijk1                     = cwrapper.prototype("void ecl_grid_get_ijk1( ecl_grid , int , int* , int* , int*)")
cfunc.get_ijk1A                    = cwrapper.prototype("void ecl_grid_get_ijk1A( ecl_grid , int , int* , int* , int*)") 
cfunc.get_xyz3                     = cwrapper.prototype("void ecl_grid_get_xyz3( ecl_grid , int , int , int , double* , double* , double*)")
cfunc.get_xyz1                     = cwrapper.prototype("void ecl_grid_get_xyz1( ecl_grid , int , double* , double* , double*)")
cfunc.get_xyz1A                    = cwrapper.prototype("void ecl_grid_get_xyz1A( ecl_grid , int , double* , double* , double*)")
cfunc.get_ijk_xyz                  = cwrapper.prototype("int  ecl_grid_get_global_index_from_xyz( ecl_grid , double , double , double , int)")
cfunc.num_lgr                      = cwrapper.prototype("int  ecl_grid_get_num_lgr( ecl_grid )")
cfunc.has_lgr                      = cwrapper.prototype("bool ecl_grid_has_lgr( ecl_grid , char* )")
cfunc.get_lgr                      = cwrapper.prototype("long ecl_grid_get_lgr( ecl_grid , char* )")
cfunc.get_cell_lgr                 = cwrapper.prototype("long ecl_grid_get_cell_lgr1( ecl_grid , int )")
cfunc.grid_value                   = cwrapper.prototype("double ecl_grid_get_property( ecl_grid , ecl_kw , int , int , int)")
cfunc.get_cell_volume              = cwrapper.prototype("double ecl_grid_get_cell_volume1( ecl_grid , int )")
cfunc.get_cell_thickness           = cwrapper.prototype("double ecl_grid_get_cell_thickness1( ecl_grid , int )")
cfunc.get_depth                    = cwrapper.prototype("double ecl_grid_get_cdepth1( ecl_grid , int )")
cfunc.fwrite_grdecl                = cwrapper.prototype("void   ecl_grid_grdecl_fprintf_kw( ecl_grid , ecl_kw , FILE , double)") 

