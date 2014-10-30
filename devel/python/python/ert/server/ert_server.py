#  Copyright (C) 2014  Statoil ASA, Norway. 
#   
#  The file 'ert_server.py' is part of ERT - Ensemble based Reservoir Tool. 
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

import sys
import threading
import json
import os

from ert.enkf import EnKFMain,RunArg,EnkfFsManager
from ert.enkf.enums import EnkfRunType, EnkfStateType, ErtImplType , EnkfVarType , RealizationStateEnum
from ert.enkf import NodeId
from ert.util import installAbortSignals

from .run_context import RunContext

class ErtCmdError(Exception):
    pass

def SUCCESS(res):
    return ["OK"] + res


def ERROR(msg , exception = None):
    return ["ERROR", msg]


class ErtServer(object):
    site_config = None

    def __init__(self , config_file , logger):
        installAbortSignals()

        self.ert_handle = None
        if os.path.exists(config_file):
            self.open( config_file )
        else:
            raise IOError("The config file:%s does not exist" % config_file)

        self.logger = logger
        self.initCmdTable()
        self.run_context = None
        self.init_fs = None
        self.run_fs = None
        self.run_count = 0



    def initCmdTable(self):
        self.cmd_table = {"STATUS" : self.handleSTATUS ,
                          "INIT_SIMULATIONS" : self.handleINIT_SIMULATIONS ,
                          "ADD_SIMULATION" : self.handleADD_SIMULATION ,
                          "SET_VARIABLE" : self.handleSET_VARIABLE ,
                          "GET_RESULT" : self.handleGET_RESULT }


    def open(self , config_file):
        self.config_file = config_file
        self.ert_handle = EnKFMain( config_file , ErtServer.site_config )
        


    def close(self):
        # More cleanup first ...
        self.ert_handle = None


    def isConnected(self):
        if self.ert_handle:
            return True
        else:
            return False


    def __del__(self):
        if self.isConnected():
            self.close()



    def evalCmd(self , cmd_expr):
        cmd = cmd_expr[0]
        func = self.cmd_table.get(cmd)

        if func:
            return func(cmd_expr[1:])
        else:
            raise ErtCmdError("The command:%s was not recognized" % cmd)


    def handleSTATUS(self , args):
        if self.isConnected():
            if self.run_context is None:
                return ["READY"]
            else:
                if self.run_context.isRunning():
                    if len(args) == 0:
                        return ["RUNNING" , self.run_context.getNumRunning() , self.run_context.getNumComplete()]
                    else:
                        iens = args[0]
                        if self.run_context.realisationComplete(iens):
                            return ["COMPLETE"]
                        else:
                            return ["RUNNING"]
                else:
                    return ["COMPLETE"]
        else:
            return ["CLOSED"]

    
    def initSimulations(self , args):
        run_size = args[0]
        init_case = str(args[1])
        run_case = str(args[2])
        
        fs_manager = self.ert_handle.getEnkfFsManager()
        self.run_fs = fs_manager.getFileSystem( run_case )
        self.init_fs = fs_manager.getFileSystem( init_case )
        fs_manager.switchFileSystem( self.run_fs )

        self.run_context = RunContext(self.ert_handle , run_size , self.run_fs  , self.run_count)
        self.run_count += 1
        return self.handleSTATUS([])


    def restartSimulations(self , args):
        return self.initSimulations(args)


    def handleINIT_SIMULATIONS(self , args):
        if len(args) == 3:
            lock = threading.Lock()
            result = []
            with lock:
                if self.run_context is None:
                    self.initSimulations( args )
                else:
                    if not self.run_context.isRunning():
                        self.restartSimulations( args )
                
                result = ["OK"]
                
            return result
        else:
            raise ErtCmdError("The INIT_SIMULATIONS command expects three arguments: [ensemble_size , init_case, run_case]")


    
    def handleGET_RESULT(self , args):
        iens = args[0]
        report_step = args[1]
        kw = str(args[2])

        ensembleConfig = self.ert_handle.ensembleConfig()
        if ensembleConfig.hasKey( kw ):
            state = self.ert_handle.getRealisation( iens )
            node = state[kw]
            gen_data = node.asGenData()
            
            fs = self.ert_handle.getEnkfFsManager().getCurrentFileSystem()
            node_id = NodeId(report_step , iens , EnkfStateType.FORECAST )
            if node.tryLoad( fs , node_id ):
                data = gen_data.getData()
                return ["OK"] + data.asList()
            else:
                raise ErtCmdError("Loading iens:%d  report:%d   kw:%s   failed" % (iens , report_step , kw))
        else:
            raise ErtCmdError("The keyword:%s is not recognized" % kw)




    def handleSET_VARIABLE(self , args):
        geo_id = args[0]
        pert_id = args[1]
        iens = args[2]
        kw = str(args[3])

        ensembleConfig = self.ert_handle.ensembleConfig()
        if ensembleConfig.hasKey(kw):
            state = self.ert_handle[iens]
            node = state[kw]
            gen_kw = node.asGenKw()
            gen_kw.setValues(args[4:])
            
            fs = self.ert_handle.getEnkfFsManager().getCurrentFileSystem()
            node_id = NodeId(0 , iens , EnkfStateType.ANALYZED )
            node.save( fs , node_id )
        else:
            raise ErtCmdError("The keyword:%s is not recognized" % kw)
            


    # ["ADD_SIMULATION" , 0 , 1 , 1 [ ["KW1" , ...] , ["KW2" , ....]]]
    def handleADD_SIMULATION(self , args):
        geo_id = args[0]
        pert_id = args[1]
        iens = args[2]
        kw_list = args[3]
        state = self.ert_handle.getRealisation( iens )
        state.addSubstKeyword( "GEO_ID" , "%s" % geo_id )
        
        elco_kw = [ l[0] for l in kw_list ]
        ens_config = self.ert_handle.ensembleConfig()

        for kw in ens_config.getKeylistFromVarType( EnkfVarType.PARAMETER ):
            if not kw in elco_kw:
                node = state[kw]
                init_id = NodeId(0 , geo_id , EnkfStateType.ANALYZED )
                run_id = NodeId(0 , iens , EnkfStateType.ANALYZED )
                node.load( self.init_fs , init_id )
                node.save( self.run_fs , run_id )
            
        for kw_arg in kw_list:
            kw = str(kw_arg[0])
            data = kw_arg[1:]
        
            node = state[kw]
            gen_kw = node.asGenKw()
            gen_kw.setValues(data)
        
            run_id = NodeId(0 , iens , EnkfStateType.ANALYZED )
            node.save( self.run_fs , run_id )

        state_map = self.run_fs.getStateMap()
        state_map[iens] = RealizationStateEnum.STATE_INITIALIZED

        self.run_context.startSimulation( iens )
        return self.handleSTATUS([])
