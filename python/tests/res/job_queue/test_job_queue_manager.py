from res.job_queue import JobStatusType, Driver,QueueDriverEnum, JobQueue, JobQueueNode, JobQueueManager
from res.enkf import ResConfig
from tests import ResTest
from ecl.util.test import TestAreaContext
import os, stat


def dummy_ok_callback(args):
    print("success {}".format(args[1]))
    with open(os.path.join(args[1],'OK'), 'w') as f:
        f.write('success')

def dummy_exit_callback(args):
    print("failure {}".format(args))
    with open('EXIT', 'w') as f:
        f.write('failure')

dummy_config = {
    "job_script" : "job_script.py",
    "num_cpu" : 1,
    "job_name" : "dummy_job_{}",
    "run_path" : "dummy_path_{}",
    "ok_callback" : dummy_ok_callback,
    "exit_callback" : dummy_exit_callback
}


def create_queue():
    driver = Driver(driver_type=QueueDriverEnum.LOCAL_DRIVER, max_running=5)
    job_queue = JobQueue(driver)
    
    with open(dummy_config["job_script"], "w") as f:
        f.write("#!/usr/bin/env python\n"\
                "with open('STATUS', 'w') as f:"\
                "   f.write('finished successfully')"\
                "\n")
    os.chmod(dummy_config["job_script"], stat.S_IRWXU |  stat.S_IRWXO |  stat.S_IRWXG )
    for i in range(10):
        os.mkdir(dummy_config["run_path"].format(i))
        job = JobQueueNode(
            job_script=dummy_config["job_script"], 
            job_name=dummy_config["job_name"].format(i), 
            run_path=os.path.realpath(dummy_config["run_path"].format(i)), 
            num_cpu=dummy_config["num_cpu"],
            status_file=job_queue.status_file, 
            ok_file=job_queue.ok_file, 
            exit_file=job_queue.exit_file,
            done_callback_function=dummy_config["ok_callback"],
            exit_callback_function=dummy_config["exit_callback"],
            callback_arguments=[{"job_number":i}, os.path.realpath(dummy_config["run_path"].format(i)), ])
        job_queue.add_job(job)
    
    return job_queue

class JobQueueManagerTest(ResTest):
    
    def test_execute_queue(self):
        
        with TestAreaContext("job_queue_manager_test") as work_area:
            job_queue = create_queue()
            manager = JobQueueManager(job_queue)
            manager.execute_queue()
            
            assert not job_queue.is_running()

            for job in job_queue.job_list:
                ok_file = os.path.realpath(os.path.join(job.run_path, "OK"))
                assert os.path.isfile( ok_file)
                with open(ok_file, 'r') as f:
                    assert f.read() == "success"
