#  Copyright (C) 2013  Statoil ASA, Norway. 
#   
#  The file 'test_run.py' is part of ERT - Ensemble based Reservoir Tool. 
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
import os.path
import subprocess
from   ert.util import TestAreaContext

class TestRun:
    __ert_cmd = "ert"
    path_prefix = None

    def __init__(self , config_file , name = None):
        if os.path.exists( config_file ):
            self.__config_file = config_file
            self.__ert_cmd = TestRun.__ert_cmd
            self.__path_prefix = TestRun.path_prefix
            
            self.args = []
            self.workflows = []
            if name:
                self.name = name
            else:
                self.name = config_file.replace("/" , ".")
                while True:
                    if self.name[0] == ".":
                        self.name = self.name[1:]
                    else:
                        break
        else:
            raise IOError("No such file or directory: %s" % config_file)


    def config_file(self):
        return self.__config_file

    #-----------------------------------------------------------------

    def set_path_prefix(self , path_prefix):
        self.__path_prefix = path_prefix
    
    def get_path_prefix(self):
        return self.__path_prefix

    path_prefix = property( get_path_prefix , set_path_prefix )

    #-----------------------------------------------------------------
    
    def get_ert_cmd(self):
        return self.__ert_cmd

    def set_ert_cmd(self , cmd):
        self.__ert_cmd = cmd

    ert_cmd = property( get_ert_cmd , set_ert_cmd)

    #-----------------------------------------------------------------

    def get_args(self):
        return self.args

    
    def add_arg(self , arg):
        self.args.append(arg)


    def get_workflows(self):
        return self.workflows

    
    def add_workflow(self , workflow):
        self.workflows.append( workflow )


    def __run(self , work_area ):
        argList = [ self.ert_cmd ]
        for arg in self.args:
            argList.append( arg )

        argList.append( self.config_file() )
        for wf in self.workflows:
            argList.append( wf )

        return subprocess.call( argList )

    
    def run(self):
        if len(self.workflows):
            with TestAreaContext(self.name , prefix = self.path_prefix) as work_area:
                print "Working in:%s" % work_area.get_cwd()
                work_area.copy_parent_directory( self.config_file() )

                status = self.__run( work_area )
        else:
            raise Exception("Must have added workflows before invoking start()")
