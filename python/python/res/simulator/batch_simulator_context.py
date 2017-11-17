import time
from collections import namedtuple
from res.server import SimulationContext
from res.enkf import NodeId, EnkfNode

Status = namedtuple("Status", "waiting pending running complete failed")

class BatchContext(SimulationContext):


    def __init__(self, result_keys, ert, fs, mask, itr):
        """
        Handle which can be used to query status and results for batch simulation.
        """
        super(BatchContext, self).__init__(ert, fs, mask, itr)
        self.result_keys = result_keys
        self.res_config = ert.resConfig( )


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
        return Status(running = self.getNumRunning(),
                      waiting = self.getNumWaiting(),
                      pending = self.getNumPending(),
                      complete = self.getNumSuccess(),
                      failed = self.getNumFailed())

    def results(self):
        """Will return the results of the simulations.

        This property will first block until all simulations have completed,
        and then it will return all the results which were configured with the
        @results when the simulator was created. The results will come as a
        list of dictionaries of arrays of double values, i.e. if the @results
        argument was:

             results = ["CMODE", "order"]

        when the simulator was created the results will be returned as:


          [  {"CMODE" : [1,2,3], "order" : [1,1,3]},
             {"CMODE" : [1,4,1], "order" : [0,7,8]},
             {"CMODE" : [6,1,0], "order" : [0,0,8] ]

        For a simulation which consist of a total of three "sub-simulations".

        """
        if self.running():
            raise RuntimeError("Simulations are still running - need to wait before gettting results")

        res = []
        nodes = [ EnkfNode(self.res_config.ensemble_config[key]) for key in self.result_keys ]
        for sim_id in range(len(self)):
            node_id = NodeId( 0, sim_id)
            d = {}
            for node in nodes:
                node.load(self.get_sim_fs(), node_id)
                gen_data = node.asGenData( )
                d[node.name()] = gen_data.getData( )
            res.append(d)

        return res
