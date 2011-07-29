#  Copyright (C) 2011  Statoil ASA, Norway. 
#   
#  The file 'ecl_queue.py' is part of ERT - Ensemble based Reservoir Tool. 
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
Implements a queue to run many ECLIPSE simulations.

The EclQueue class is a small wrapper around the JobQueue class, which
is specialized to only submit ECLIPSE simulations. 

The EclQueue class should be used when you have more simulations than
you can perform in parallell; when using used in combination with LSF
the whole concept of a private queue is somewhat superfluos. 

The queue is based on the use of 'driver' to communicate with the low
level systems for running the simulations. The driver must be
instantiated before you can create a queue. The driver is implemented
in the driver module in ert.job_queue.driver.
"""
import os.path

from   ert.job_queue.queue  import JobQueue
import ert.job_queue.driver as queue_driver
import libecl
from   ecl_default import default
import ecl_util


class EclQueue( JobQueue ):
    def __init__(self , 
                 driver = None , 
                 driver_type = default.driver_type, 
                 driver_options = None, 
                 ecl_version = default.ecl_version , 
                 ecl_cmd = default.ecl_cmd , 
                 max_running = 0,
                 size = 0):

        """
        Short doc.

           max_running: The maximum number of jobs the queue can run
              concurrently. The default value max_running=0 means that
              the queue can take an unlimited number of jobs.
        
        When it comes to the size argument there are two alternatives:

           size = 0: That means that you do not tell the queue in
              advance how many jobs you have. The queue will just run
              all the jobs you add, but you have to inform the queue
              in some way that all jobs have been submitted, and no
              more will be coming. To achieve this you should call the
              submit_complete() method when all jobs are submitted.

           size > 0: The queue will now exactly how many jobs to run,
              and will continue until this number of jobs have
              completed.
        """

        self.ecl_version = ecl_version
        self.ecl_cmd     = ecl_cmd
        if driver is None:
            if driver_options is None:
                driver_options = default.driver_options[ driver_type ]
            driver = queue_driver.Driver( driver_type , max_running = max_running , options = driver_options )
        JobQueue.__init__( self , driver , self.ecl_cmd , size = size)

        
    def add_job( self , data_file):
        (path_base , ext) = os.path.splitext( data_file )
        (run_path , base) = os.path.split( path_base )
        
        argv = [ self.ecl_version , path_base , "%s" % ecl_util.get_num_cpu( data_file )]
        return JobQueue.add_job( self , self.ecl_cmd , run_path , base , argv)

