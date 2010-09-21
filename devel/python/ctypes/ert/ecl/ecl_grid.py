import ctypes
from   ert.cwrap.cwrap       import *
import ert.util.stringlist
import fortio



class EclGrid(object):

    def __new__(cls , filename, lgr = None , parent = None):
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
    def active( self ):
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
        if active_index:
            set_count += 1
        if global_index:
            set_count += 1
        if ijk:
            set_count += 1
            
        if not set_count == 1:
            sys.exit("The function get_xyz() requires that exactly one of the kewyord arguments active_index, global_index, ijk be set")
        
        if active_index:
            global_index = cfunc.get_global_index1A( self , active_index )
        elif ijk:
            global_index = cfunc.get_global_index3( self , ijk[0] , ijk[1] , ijk[2])
        
        return global_index
                 

    def get_active_index( self , ijk = None , global_index = None):
        gi = self.__global_index( global_index = global_index , ijk = ijk)
        return cfunc.get_active_index1( self , gi)


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


    def grid_value( self , ecl_kw , i,j,k):
        return cfunc.grid_value( self , ecl_kw , i , j , k)




# 1. Loading the necessary C-libraries.
ctypes.CDLL("libz.so"      , ctypes.RTLD_GLOBAL)
ctypes.CDLL("libblas.so"   , ctypes.RTLD_GLOBAL)
ctypes.CDLL("liblapack.so" , ctypes.RTLD_GLOBAL)
ctypes.CDLL("libutil.so" , ctypes.RTLD_GLOBAL)
libecl  = ctypes.CDLL("libecl.so"  , ctypes.RTLD_GLOBAL)


# 2. Creating a wrapper object around the libecl library, 
#    registering the type map : ecl_kw <-> EclKW
cwrapper = CWrapper( libecl )
cwrapper.registerType( "ecl_grid" , EclGrid )

# 3. Installing the c-functions used to manipulate ecl_kw instances.
#    These functions are used when implementing the EclKW class, not
#    used outside this scope.
cfunc = CWrapperNameSpace("ecl_grid")


cfunc.fread_alloc                  = cwrapper.prototype("long ecl_grid_load_case( char* )")
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
