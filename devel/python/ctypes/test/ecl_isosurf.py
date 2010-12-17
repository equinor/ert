#!/prog/sdpsoft/python2.4/bin/python
import sys
import os.path
import ert.ecl.ecl_util as ecl_util
import ert.ecl as ecl
import ert.util.tvector
from   ert.util.lookup_table import LookupTable


# This function will scan the vertical column specified by (i,j) and
# collect the pairs ( value(kw) , depth ) in a lookuptable. When the
# full column has been scanned it will lookup the value @level in the
# lookuptable and return the corresponding depth, based on linear
# interpolation.
#
# If @level is out of bounds - i.e. it is above the highest value
# found in the column, or below the lowest value, the behaviour will
# depend on whether we are searching for gas levels or water levels,
# if we aware searching for gas the behaviour is as follows:
# 
#  o If @level is above the found maximum level we return reservoir
#    top.
#  o If @level is below the found minimum level we return reservoir
#    bottom.
#
# When searching for water levels the behaviour os opposite.
#
# If the only one or zero active cells is founc in the column, the
# value None is returned.
#
# 
# Caveat: Non unique depth profile
#

def iso_level( grid , kw , level , i , j ):
    depth_value = LookupTable()
    klist = range(grid.nz) 
    
    for k in klist:
        if grid.active( ijk = (i,j,k) ):
            depth_value.append( grid.grid_value( kw , i,j,k) , grid.depth( ijk = (i,j,k)))

    if depth_value.size > 1:
        if depth_value.arg_max > level and depth_value.arg_min < level:
            depth = depth_value.interp( level ) 
            k = grid.locate_depth( depth , i , j )
            return (depth , k)
        else:
            return (None , 0)
    else:
        return (None , 0)



def write_surface( grid , kw , output_fmt , level ):
    output_file = output_fmt % level
    (path , file) = os.path.split( output_file )
    if path:
        if not os.path.exists( path ):
            print "Creating directory: %s" % path
            os.makedirs( path )
    fileH = open( output_file , "w")
    for i in range(grid.nx):
        for j in range(grid.ny):
            (depth , k) = iso_level( grid , kw , level ,i,j)
            if depth:
                if k >= 0:
                    (utm_x , utm_y , tmp) = grid.get_xyz( ijk=(i,j,k) )
                    fileH.write("%12.7f  %12.7f  %12.7f\n" % (utm_x , utm_y , depth))
                else:
                    sys.stderr.write("Warning depth mapping problem in i=%d,j=%d level:%g\n" % (i,j,level))
    fileH.close()




def load_input( arglist ):
    input_file = arglist[0]
    file_type = ecl_util.get_file_type( input_file )
    if file_type == ecl_util.ECL_UNIFIED_RESTART_FILE:
        try:
            report_step = int( arglist[1] )
        except:
            sys.exit("When using unified restart file you must specify report number in the next argument")

        if os.path.exists( input_file ):
            ecl_file = ecl.EclFile.restart_block( input_file , report_step = report_step )
        else:
            sys.exit("Input file:%s not found." % input_file)
        arg_offset = 2
    elif file_type == ecl_util.ECL_RESTART_FILE:
        if os.path.exists( input_file ):
            ecl_file = ecl.EclFile( input_file )
        else:
            sys.exit("Input file:%s not found." % input_file)
    else:
        sys.exit("Input argument:%s must be restart file." % input_file)
        arg_offset = 1
    
    (path_base , ext) = os.path.splitext( input_file )
    grid = ecl.EclGrid( path_base )

    phase = arglist[ arg_offset ].upper()
    if phase in ["SWAT" , "SGAS"]:
        kw = ecl_file.iget_named_kw( phase , 0 )
    else:
        sys.exit("Must give phase : SWAT / SGAS ")
        
    output_fmt = arglist[ arg_offset + 1 ]
    level_list = []
    for level in arglist[arg_offset + 2:]:
        level_list.append( float( level ) )
        
    return (grid , kw , output_fmt , level_list )




restart_file = "data/eclipse/case/ECLIPSE.UNRST"

sys.argv = [None , restart_file , "40" , "SWAT" , "/tmp/iso_swat_%f" , "0.30" , "0.40" , "0.50" , "0.60" , "0.70"]
(grid , kw , output_fmt , level_list) = load_input( sys.argv[1:] )

for level in level_list:
    write_surface( grid , kw , output_fmt , level )
