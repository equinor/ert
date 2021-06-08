import time
from collections import namedtuple
from res.enkf import NodeId, EnkfNode
import logging
import numpy as np
from .simulation_context import SimulationContext

Status = namedtuple("Status", "waiting pending running complete failed")


class BatchContext(SimulationContext):
    def __init__(self, result_keys, ert, fs, mask, itr, case_data):
        """
        Handle which can be used to query status and results for batch simulation.
        """
        super(BatchContext, self).__init__(ert, fs, mask, itr, case_data)
        self.result_keys = result_keys
        self.res_config = ert.resConfig()

    def join(self):
        """
        Will block until the simulation is complete.
        """
        while self.running():
            time.sleep(1)

    def running(self):
        return self.isRunning()

    @property
    def status(self):
        """
        Will return the state of the simulations.
        """
        return Status(
            running=self.getNumRunning(),
            waiting=self.getNumWaiting(),
            pending=self.getNumPending(),
            complete=self.getNumSuccess(),
            failed=self.getNumFailed(),
        )

    def results(self):
        """Will return the results of the simulations.

        Observe that this function will raise RuntimeError if the simulations
        have not been completed. To be certain that the simulations have
        completed you can call the join() method which will block until all
        simulations have completed.

        The function will return all the results which were configured with the
        @results when the simulator was created. The results will come as a
        list of dictionaries of arrays of double values, i.e. if the @results
        argument was:

             results = ["CMODE", "order"]

        when the simulator was created the results will be returned as:


          [ {"CMODE" : [1,2,3], "order" : [1,1,3]},
            {"CMODE" : [1,4,1], "order" : [0,7,8]},
            None,
            {"CMODE" : [6,1,0], "order" : [0,0,8]} ]

        For a simulation which consist of a total of four simulations, where the
        None value indicates that the simulator was unable to compute a request.
        The order of the list corresponds to case_data provided in the start
        call.

        """
        if self.running():
            raise RuntimeError(
                "Simulations are still running - need to wait before gettting results"
            )

        res = []
        nodes = [
            EnkfNode(self.res_config.ensemble_config[key]) for key in self.result_keys
        ]
        for sim_id in range(len(self)):
            node_id = NodeId(0, sim_id)
            if not self.didRealizationSucceed(sim_id):
                logging.error(
                    "Simulation %d (node %s) failed." % (sim_id, str(node_id))
                )
                res.append(None)
                continue
            d = {}
            for node in nodes:
                node.load(self.get_sim_fs(), node_id)
                data = node.asGenData().getData()
                d[node.name()] = np.array(data)
            res.append(d)

        return res
