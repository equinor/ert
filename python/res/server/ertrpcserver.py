import os, sys
from threading import Lock

try:
    from SimpleXMLRPCServer import SimpleXMLRPCServer
    from xmlrpclib import Fault
except ImportError:
    from xmlrpc.server import SimpleXMLRPCServer
    from xmlrpc.client import Fault


import warnings
from res.util import ResVersion
from res.enkf import EnKFMain, NodeId, ResConfig
from ecl.util.util import BoolVector
from res.enkf.config import CustomKWConfig
from res.enkf.data import EnkfNode, CustomKW
from res.enkf.enums import RealizationStateEnum, EnkfVarType, ErtImplType
from res.server import SimulationContext
from res.server.ertrpcclient import FAULT_CODES


def checkRealizationState(state):
    return state == RealizationStateEnum.STATE_INITIALIZED or state == RealizationStateEnum.STATE_HAS_DATA




class Session(object):
    def __init__(self):
        self.geo_case = None
        """ :type: str """

        self.simulation_context = None
        """ :type: SimulationContext """

        self.batch_number = 0
        self.lock = Lock()

INVERSE_FAULT_CODES = {value:key for key, value in FAULT_CODES.items()}

def createFault(error, message):
    error_code = INVERSE_FAULT_CODES[error]
    return Fault(error_code, message)

class ErtRPCServer(SimpleXMLRPCServer):
    def __init__(self, enkf_main, host="localhost", port=0, log_requests=False, verbose_queue=False):
        SimpleXMLRPCServer.__init__(self, (host, port), allow_none=True, logRequests=log_requests)
        self.ert_version = ResVersion( )
        self._host = host
        self._verbose_queue = verbose_queue
        # https: server.socket = ssl.wrap_socket(srv.socket, ...)

        self._init_enkf_main(enkf_main)

        self._session = Session()

        self.register_function(self.ertVersion)
        self.register_function(self.getTimeMap)
        self.register_function(self.isRunning)
        self.register_function(self.isInitializationCaseAvailable)
        self.register_function(self.startSimulationBatch)
        self.register_function(self.addSimulation)
        self.register_function(self.isRealizationFinished)
        self.register_function(self.didRealizationSucceed)
        self.register_function(self.didRealizationFail)
        self.register_function(self.getGenDataResult)
        self.register_function(self.getCustomKWResult)
        self.register_function(self.isCustomKWKey)
        self.register_function(self.isGenDataKey)
        self.register_function(self.prototypeStorage)
        self.register_function(self.storeGlobalData)
        self.register_function(self.storeSimulationData)


    def _init_enkf_main(self, enkf_main):
        if enkf_main is None:
            raise ValueError("Expected enkf_main to be "
                    "provided upon initialization of ErtRPCServer. "
                    "Received enkf_main=%r"
                    % enkf_main
                    )

        if isinstance(enkf_main, EnKFMain):
            self._enkf_main   = enkf_main
            self._res_config  = enkf_main.resConfig
            self._config_file = enkf_main.getUserConfigFile()

        else:
            if sys.version_info[0] == 2:
               check = isinstance(enkf_main, basestring)
            else:
               check = isinstance(enkf_main, str)
            if check:
               if not os.path.exists(enkf_main):
                   raise IOError("No such configuration file: %s" % enkf_main)

               warnings.warn("Providing a configuration file as the argument "
                    "to ErtRPCServer is deprecated. Please use the "
                    "optional argument res_config to provide an "
                    "configuration!",
                    DeprecationWarning
                    )

               self._config_file = enkf_main
               self._res_config  = ResConfig(self._config_file)
               self._enkf_main   = EnKFMain(self._res_config)

            else:
               raise TypeError("Expected enkf_main to be an instance of "
                    "EnKFMain, received: %r" % enkf_main)

    @property
    def port(self):
        return self.server_address[1]

    @property
    def host(self):
        return self._host

    @property
    def ert(self):
        return self._enkf_main

    def start(self):
        self.serve_forever()

    def stop(self):
        if self._session.simulation_context is not None:
            if self._session.simulation_context.isRunning():
                self._session.simulation_context._queue_manager.stop_queue()
        self.shutdown()
        self.server_close()
        self._enkf_main = None

    def ertVersion(self):
        return self.ert_version.versionTuple()

    def getTimeMap(self, target_case_name):
        enkf_fs_manager = self.ert.getEnkfFsManager()
        enkf_fs = enkf_fs_manager.getFileSystem(target_case_name)
        time_map = enkf_fs.getTimeMap()
        return [time_step.datetime() for time_step in time_map]

    def isRunning(self):
        if self._session.simulation_context is not None:
            return self._session.simulation_context.isRunning()
        return False

    def isRealizationFinished(self, iens):
        if self._session.simulation_context is None:
            raise createFault(UserWarning, "The simulation batch has not been initialized")

        if self._session.simulation_context.isRealizationQueued(iens):
            return self._session.simulation_context.isRealizationFinished(iens)
        return False

    def didRealizationSucceed(self, iens):
        if self._session.simulation_context is not None and self._session.simulation_context.isRealizationQueued(iens):
            return self._session.simulation_context.didRealizationSucceed(iens)
        return False

    def didRealizationFail(self, iens):
        if self._session.simulation_context is not None and self._session.simulation_context.isRealizationQueued(iens):
            return self._session.simulation_context.didRealizationFail(iens)
        return False


    def isInitializationCaseAvailable(self):
        return self._session.geo_case is not None


    def startSimulationBatch(self, initialization_case_name, result_case_name, simulation_count):
        fs_manager = self.ert.getEnkfFsManager()
        init_fs = fs_manager.getFileSystem(initialization_case_name)
        sim_fs = fs_manager.getFileSystem(result_case_name)
        mask = BoolVector( initial_size = simulation_count, default_value = True )
        with self._session.lock:
            if not self.isRunning():
                self._session.simulation_context = None
                self._session.geo_case = initialization_case_name

                self.ert.addDataKW("<WPRO_RUN_COUNT>", str(self._session.batch_number))
                self.ert.addDataKW("<ELCO_RUN_COUNT>", str(self._session.batch_number))
                self._session.simulation_context = SimulationContext(self.ert, sim_fs, mask , self._session.batch_number, verbose=self._verbose_queue)
                self._session.batch_number += 1


    def addSimulation(self, geo_id, pert_id, iens, keywords):
        if not self.isRunning():
            raise createFault(UserWarning, "The server is not ready to receive simulations. Have you called startSimulationBatch() first?")

        if self._session.simulation_context.isRealizationQueued(iens):
            raise createFault(UserWarning, "Simulation with id: '%d' is already running." % iens)

        sim_fs = self._session.simulation_context.get_sim_fs( )
        self._initializeRealization(sim_fs, geo_id, iens, keywords)

        self._session.simulation_context.addSimulation(iens, geo_id)


    def _initializeRealization(self, sim_fs, geo_id, iens, keywords):
        ens_config = self.ert.ensembleConfig()

        # Copy all parameter all parameter values which are not given by @keywords from the
        # the initialization case to the target case.

        geo_case_fs = self.ert.getEnkfFsManager().getFileSystem(self._session.geo_case)
        for kw in ens_config.getKeylistFromVarType(EnkfVarType.PARAMETER):
            if not kw in keywords:
                config_node = ens_config[kw]
                data_node = EnkfNode( config_node )
                init_id = NodeId(0, geo_id)
                run_id = NodeId(0, iens)
                data_node.load(geo_case_fs , init_id)
                data_node.save(sim_fs, run_id)

        # All the values supplied externally by the keywords list will be written
        # directly into the result case.
        for key, values in keywords.items():
            config_node = ens_config[key]
            data_node = EnkfNode( config_node )

            gen_kw = data_node.asGenKw()
            gen_kw.setValues(values)

            run_id = NodeId(0, iens)
            data_node.save(sim_fs, run_id)

        sim_fs.fsync()
        state_map = sim_fs.getStateMap()
        state_map[iens] = RealizationStateEnum.STATE_INITIALIZED


    def getGenDataResult(self, target_case_name, iens, report_step, keyword):
        ensemble_config = self.ert.ensembleConfig()

        if not self.isRealizationFinished(iens):
            raise createFault(UserWarning, "The simulation with id: %d is still running." % iens)

        if keyword in ensemble_config:
            enkf_config_node = self.ert.ensembleConfig().getNode(keyword)
            node = EnkfNode(enkf_config_node)

            if not node.getImplType() == ErtImplType.GEN_DATA:
                raise createFault(UserWarning, "The keyword is not a GenData keyword.")

            gen_data = node.asGenData()

            fs = self.ert.getEnkfFsManager().getFileSystem(target_case_name)
            node_id = NodeId(report_step, iens)
            if node.tryLoad(fs, node_id):
                data = gen_data.getData()
                return data.asList()
            else:
                raise createFault(UserWarning, "Unable to load data for iens: %d report_step: %d kw: %s for case: %s" % (iens, report_step, keyword, target_case_name))
        else:
            raise createFault(KeyError, "The keyword: %s is not recognized" % keyword)


    def getCustomKWResult(self, target_case_name, iens, keyword):
        ensemble_config = self.ert.ensembleConfig()

        if not self.isRealizationFinished(iens):
            raise createFault(UserWarning, "The simulation with id: %d is still running." % iens)

        if keyword in ensemble_config:
            enkf_config_node = self.ert.ensembleConfig().getNode(keyword)
            node = EnkfNode(enkf_config_node)

            if not node.getImplType() == ErtImplType.CUSTOM_KW:
                raise createFault(UserWarning, "The keyword is not a CustomKW keyword.")

            custom_kw = node.asCustomKW()

            fs = self.ert.getEnkfFsManager().getFileSystem(target_case_name)
            node_id = NodeId(0, iens)
            if node.tryLoad(fs, node_id):
                config = custom_kw.getConfig()
                result = {}
                for key in config.getKeys():
                    result[key] = custom_kw[key]
                return result
            else:
                raise createFault(UserWarning, "Unable to load data for iens: %d kw: %s for case: %s" % (iens, keyword, target_case_name))
        else:
            raise createFault(KeyError, "The keyword: %s is not recognized" % keyword)

    def getFailedCount(self):
        if self._session.simulation_context is not None:
            return self._session.simulation_context.getNumFailed()
        else:
            return 0

    def getRunningCount(self):
        if self._session.simulation_context is not None:
            return self._session.simulation_context.getNumRunning()
        else:
            return 0

    def getSuccessCount(self):
        if self._session.simulation_context is not None:
            return self._session.simulation_context.getNumSuccess()
        else:
            return 0

    def getWaitingCount(self):
        if self._session.simulation_context is not None:
            return self._session.simulation_context.getNumWaiting()
        else:
            return 0

    def getBatchNumber(self):
        return self._session.batch_number

    def isCustomKWKey(self, key):
        ensemble_config = self.ert.ensembleConfig()
        return key in ensemble_config.getKeylistFromImplType(ErtImplType.CUSTOM_KW)

    def isGenDataKey(self, key):
        ensemble_config = self.ert.ensembleConfig()
        return key in ensemble_config.getKeylistFromImplType(ErtImplType.GEN_DATA)


    def prototypeStorage(self, group_name, storage_definition):
        ensemble_config = self.ert.ensembleConfig()

        if group_name in ensemble_config.getKeylistFromImplType(ErtImplType.CUSTOM_KW):
            raise createFault(UserWarning, "The CustomKW with group name: '%s' already exist!" % group_name)

        converted_definition = {}
        for key, value in storage_definition.items():
            if value == "str":
                converted_definition[key] = str
            elif value == "float":
                converted_definition[key] = float
            else:
                raise createFault(TypeError, "Unknown type: '%s' for key '%s'" % (value, key))

        enkf_config_node = ensemble_config.addDefinedCustomKW(group_name, converted_definition)
        self.ert.addNode(enkf_config_node)

    def storeGlobalData(self, target_case_name, group_name, keyword, value):
        fs = self.ert.getEnkfFsManager().getFileSystem(target_case_name)
        enkf_config_node = self.ert.ensembleConfig().getNode(group_name)
        enkf_node = EnkfNode(enkf_config_node)
        self._updateCustomKWConfigSet(fs, enkf_config_node)

        realizations = fs.realizationList(RealizationStateEnum.STATE_INITIALIZED | RealizationStateEnum.STATE_HAS_DATA)

        for realization_number in realizations:
            self._storeData(enkf_node, fs, group_name, keyword, value, realization_number)

    def storeSimulationData(self, target_case_name, group_name, keyword, value, sim_id):
        fs = self.ert.getEnkfFsManager().getFileSystem(target_case_name)
        enkf_config_node = self.ert.ensembleConfig().getNode(group_name)
        enkf_node = EnkfNode(enkf_config_node)
        self._updateCustomKWConfigSet(fs, enkf_config_node)

        self._storeData(enkf_node, fs, group_name, keyword, value, sim_id)

    def _updateCustomKWConfigSet(self, fs, enkf_config_node):
        ckwcs = fs.getCustomKWConfigSet()
        ckwcs.addConfig(enkf_config_node.getCustomKeywordModelConfig())


    def _storeData(self, enkf_node, fs, group_name, keyword, value, realization_number):
        node_id = NodeId(0, realization_number)
        enkf_node.tryLoad(fs, node_id)  # Fetch any data from previous store calls
        custom_kw = enkf_node.asCustomKW()
        custom_kw[keyword] = value

        if not enkf_node.save(fs, node_id):
            raise createFault(UserWarning, "Unable to store data for group: '%s' and key: '%s' into realization: '%d'" % (group_name, keyword, realization_number))








