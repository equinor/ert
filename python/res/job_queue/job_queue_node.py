from cwrap import BaseCClass
from ecl.util.util import StringList
from res import ResPrototype
from res.job_queue import JobStatusType
from threading import Thread

import time


class JobQueueNode(BaseCClass):
    TYPE_NAME = "job_queue_node"
    
    _alloc = ResPrototype("void* job_queue_node_alloc_python(char*,"\
                                            "char*,"\
                                            "char*,"\
                                            "int, "\
                                            "stringlist,"\
                                            "int, "\
                                            "char*,"\
                                            "char*,"\
                                            "char*"\
                                            ")", bind=False)
    _free = ResPrototype("void job_queue_node_free(job_queue_node)")
    _submit = ResPrototype("int job_queue_node_submit_simple(job_queue_node, driver)")
    _kill = ResPrototype("bool job_queue_node_kill_simple(job_queue_node, driver)")
    
    _get_status = ResPrototype("int job_queue_node_get_status(job_queue_node)")
    _update_status = ResPrototype("bool job_queue_node_update_status_simple(job_queue_node, driver)")


    def __init__(self,job_script, job_name, run_path, num_cpu, 
                    status_file, ok_file, exit_file, 
                    done_callback_function, exit_callback_function, callback_arguments):
        self.done_callback_function = done_callback_function
        self.exit_callback_function = exit_callback_function
        self.callback_arguments = callback_arguments
    
        argc = 1
        argv = StringList()
        argv.append(run_path)
        self.started = False
        self.run_path = run_path
        c_ptr = self._alloc(job_name, run_path, job_script, argc, argv, num_cpu,
                                ok_file, status_file, exit_file, 
                                None, None, None, None)

        if c_ptr is not None:
            super(JobQueueNode, self).__init__(c_ptr)
        else:
            raise ValueError("Unable to create job node object")
    
    def free(self):
        self._free()

    @property
    def status(self):
        return self._get_status()

    def submit(self, driver):
        self._submit(driver)

    def run_done_callback(self):
        return self.done_callback_function(self.callback_arguments)

    def run_exit_callback(self):
        return self.exit_callback_function(self.callback_arguments)
    
    def is_running(self):
        return (self.status ==  JobStatusType.JOB_QUEUE_PENDING or 
                self.status == JobStatusType.JOB_QUEUE_SUBMITTED or
                self.status == JobStatusType.JOB_QUEUE_RUNNING  or
                self.status == JobStatusType.JOB_QUEUE_UNKNOWN) # dont stop monitoring if LSF commands are unavailable
    
    def job_monitor(self, driver):

        self._submit(driver)
        self.update_status(driver)

        while self.is_running():
            time.sleep(1)
            self.update_status(driver)

        if self.status == JobStatusType.JOB_QUEUE_DONE:
            self.run_done_callback()
        elif self.status == JobStatusType.JOB_QUEUE_EXIT:
            self.run_exit_callback()
        elif self.status == JobStatusType.JOB_QUEUE_WAITING:
            self.started = False

    def run(self, driver):
        self.started = True
        x = Thread(target=self.job_monitor, args=(driver, ))
        x.start()
        return x

    def stop(self,  driver):
        self._kill(driver)

    def update_status(self, driver):
        if self.status != JobStatusType.JOB_QUEUE_WAITING:
            self._update_status(driver)

