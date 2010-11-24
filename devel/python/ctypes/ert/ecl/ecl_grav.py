from    ert.cwrap.cwrap       import *
import  libecl

from   ecl_kw                import EclKW , ECL_INT_TYPE
from   ecl_grid              import EclGrid



def phase_deltag( xyz , grid , aquifern , sat1 , rho1 , porv1 , sat2 , rho2 , porv2):
    if not aquifern:
        aquifern = EclKW.new("AQUIFERN" , rho1.size , ECL_INT_TYPE)
        aquifern.assign( 0 )
    
        # Something like this:
        # aquifern = ctypes.cast(None , ctypes.POINTER(ctypes.c_long)) 

    
    # Observe that the function cfunc.phase_deltag() is called with
    # directly with c_ptr pointer values, and that the function is
    # prototyped with long arguments instead of ecl_grid and
    # ecl_kw. For some strange reason some of the underlying C pointer
    # values were molested when reaching the C function. This is
    # avoided by using the pointer value directly. Maybe this is a
    # Python/ctypes bug in the current 2.4 implementation??

    return cfunc.phase_deltag( xyz[0] , xyz[1] , xyz[2] , 
                               grid.c_ptr , aquifern.c_ptr , 
                               sat1.c_ptr , rho1.c_ptr , porv1.c_ptr , 
                               sat2.c_ptr , rho2.c_ptr , porv2.c_ptr )
    

# 1. All restart files should have water, i.e. the SWAT keyword. 
# 2. All phases present in the restart file should also be present as densities, 
#    in addition the model must contain one additional phase - which should have a density.
# 3. The restart files can never contain oil saturation.

def deltag( xyz , grid , init_file , restart_file1 , restart_file2):
    if init_file.has_kw( "AQUIFERN" ):
        aquifern = init_file.iget_named_kw( "AQUIFERN" , 0 )
    else:
        aquifern = None
        
    swat1 = restart_file1.iget_named_kw( "SWAT" , 0)
    swat2 = restart_file2.iget_named_kw( "SWAT" , 0)
    phase_list = [ ( swat1 , swat2) ]

    if restart_file1.has_kw( "SGAS" ):
        # This is a three phase Water / Gas / Oil system 
        sgas1 = restart_file1.iget_named_kw( "SGAS" , 0 )
        sgas2 = restart_file2.iget_named_kw( "SGAS" , 0 )

        soil1 = 1 - (sgas1 + swat1)
        soil2 = 1 - (sgas1 + swat1)
        soil1.name = "SOIL"
        soil2.name = "SOIL"
        phase_list += [ (sgas1 , sgas2),
                        (soil1 , soil2) ]
    else:
        # This is a two phase Water / Gas system
        sgas1 = 1 - swat1
        sgas2 = 1 - swat2
        sgas1.name = "SGAS"
        sgas2.name = "SGAS"
        phase_list += [ (sgas1 , sgas2) ]

    porv1 = restart_file1.iget_named_kw( "RPORV" , 0 )
    porv2 = restart_file2.iget_named_kw( "RPORV" , 0 )
    
    deltag = 0
    for (sat1,sat2) in phase_list:
        rho_name = "%s_DEN" % sat1.name[1:]   
        rho1 = restart_file1.iget_named_kw( rho_name , 0 )
        rho2 = restart_file2.iget_named_kw( rho_name , 0 )
        deltag += phase_deltag( xyz , grid , aquifern , sat1 , rho1 , porv1 , sat2 , rho2 , porv2)
    return deltag




cwrapper           = CWrapper( libecl.lib )
cfunc              = CWrapperNameSpace("ecl_grav")
#cfunc.phase_deltag = cwrapper.prototype("double ecl_grav_phase_deltag( double, double ,double , ecl_grid , ecl_kw , ecl_kw , ecl_kw , ecl_kw , ecl_kw , ecl_kw , ecl_kw)")
cfunc.phase_deltag = cwrapper.prototype("double ecl_grav_phase_deltag( double, double ,double , long , long , long , long , long , long , long , long)")
cfunc.phase_test_deltag = cwrapper.prototype("double ecl_grav_phase_test_deltag( double , double , double , ecl_grid , ecl_kw , ecl_kw , ecl_kw , ecl_kw)")

