#!/prog/sdpsoft/python2.4/bin/python
import os
import shutil
import os.path
import ert.ecl       as ecl
import ert.job_queue as job_queue


def copy_case( target_path , src_files):
    if not os.path.exists( target_path ):
        os.makedirs( target_path )

    print "Creating simulation directory:%s" % target_path
    for file in src_files:
        if os.path.isfile( file ):
            shutil.copy( file , target_path )
        elif os.path.isdir( file ):
            (path , base) = os.path.split( file )
            if not os.path.exists( "%s/%s" % (target_path , base)):
                shutil.copytree( file , "%s/%s" % (target_path , base) )
        else:
            sys.exit("Error")
            


src_files     = ["data/eclipse/case/ECLIPSE.DATA" , "data/eclipse/case/include"]
run_path_fmt  = "tmp/simulations/run%d"


for case_nr in range( 10 ):
    copy_case( run_path_fmt % case_nr , src_files )

driver = job_queue.LocalDriver()


