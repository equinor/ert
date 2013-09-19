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
from   ert.util import TestAreaContext

class TestRun:
    ert_cmd = "ert"

    def __init__(self , config_file , name = None):
        if os.path.exists( config_file ):
            self._config_file = config_file
            self._ert_cmd = TestRun.ert_cmd
            self.args = []
            self.workflows = []
            if name:
                self.name = name
            else:
                self.name = config_file.replace("/" , ".")
        else:
            raise IOError("No such file or directory: %s" % config_file)


    def config_file(self):
        return self._config_file

    
    def get_cmd(self):
        return self._ert_cmd


    def set_cmd(self , cmd):
        self._ert_cmd = cmd
        

    def get_args(self):
        return self.args

    
    def add_arg(self , arg):
        self.args.append(arg)


    def get_workflows(self):
        return self.workflows

    
    def add_workflow(self , workflow):
        self.workflows.append( workflow )

    
    def start(self):
        if len(self.workflows):
            pass
            with TestAreaContext(self.name , True) as work_area:
                print work_area.get_cwd()
                pass
        else:
            raise Exception("Must have added workflows before invoking start()")
