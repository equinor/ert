from res.job_queue import JobStatusType, Driver, QueueDriverEnum, JobQueue, JobQueueNode
from tests import ResTest
from tests.utils import wait_until
from ecl.util.test import TestAreaContext
import os, stat, time
from threading import BoundedSemaphore

def dummy_ok_callback(args):
    print(args)

def dummy_exit_callback(args):
    print(args)

dummy_config = {
    "job_script" : "job_script.py",
    "num_cpu" : 1,
    "job_name" : "dummy_job_{}",
    "run_path" : "dummy_path_{}",
    "ok_callback" : dummy_ok_callback,
    "exit_callback" : dummy_exit_callback
}

simple_script = """#!/usr/bin/env python
print('hello')
"""

never_ending_script = """#!/usr/bin/env python
import time
while True:
    time.sleep(0.5)
"""

failing_script = """#!/usr/bin/env python
import sys
sys.exit(1)
"""

def create_queue(script, max_submit=1):
    driver = Driver(driver_type=QueueDriverEnum.LOCAL_DRIVER, max_running=5)
    job_queue = JobQueue(driver, max_submit=max_submit)
    with open(dummy_config["job_script"], "w") as f:
        f.write(script)
    os.chmod(dummy_config["job_script"], stat.S_IRWXU |  stat.S_IRWXO |  stat.S_IRWXG )
    for i in range(10):
        os.mkdir(dummy_config["run_path"].format(i))
        job = JobQueueNode(
            job_script=dummy_config["job_script"],
            job_name=dummy_config["job_name"].format(i),
            run_path=dummy_config["run_path"].format(i),
            num_cpu=dummy_config["num_cpu"],
            status_file=job_queue.status_file,
            ok_file=job_queue.ok_file,
            exit_file=job_queue.exit_file,
            done_callback_function=dummy_config["ok_callback"],
            exit_callback_function=dummy_config["exit_callback"],
            callback_arguments=[{"job_number":i}])

        job_queue.add_job(job)
    
    return job_queue

def start_all(job_queue, sema_pool):
    job = job_queue.fetch_next_waiting()
    while(job is not None):
        job.run(job_queue.driver, sema_pool, job_queue.max_submit)
        job = job_queue.fetch_next_waiting()
    

class JobQueueTest(ResTest):

    def testStatusEnum(self):
        source_path = "lib/include/ert/job_queue/job_status.hpp"
        self.assertEnumIsFullyDefined(JobStatusType, "job_status_type", source_path)

    def test_kill_jobs(self):
        with TestAreaContext("job_queue_test_kill") as work_area:
            job_queue = create_queue(never_ending_script)

            assert job_queue.queue_size == 10
            assert job_queue.is_active()
            
            pool_sema = BoundedSemaphore(value=10)
            start_all(job_queue, pool_sema)

            # make sure never ending jobs are running
            wait_until(
                lambda: self.assertTrue(job_queue.is_active())
            )

            for job in job_queue.job_list:
                job.stop()

            wait_until(
                lambda: self.assertFalse(job_queue.is_active())
            )
            
            for job in job_queue.job_list:
                assert job.status == JobStatusType.JOB_QUEUE_IS_KILLED
            
            for job in job_queue.job_list:
                job.wait_for()

    def test_add_jobs(self):
        with TestAreaContext("job_queue_test_add") as work_area:
            job_queue = create_queue(simple_script)

            assert job_queue.queue_size == 10
            assert job_queue.is_active()
            assert job_queue.fetch_next_waiting() is not None

            pool_sema = BoundedSemaphore(value=10)
            start_all(job_queue, pool_sema)

            for job in job_queue.job_list:
                job.stop()

            wait_until(
                lambda: self.assertFalse(job_queue.is_active())
            )

            for job in job_queue.job_list:
                job.wait_for()

    def test_failing_jobs(self):
        with TestAreaContext("job_queue_test_add") as work_area:
            job_queue = create_queue(failing_script, max_submit=1)

            assert job_queue.queue_size == 10
            assert job_queue.is_active()

            pool_sema = BoundedSemaphore(value=10)            
            start_all(job_queue, pool_sema)

            wait_until(
                func=(lambda: self.assertFalse(job_queue.is_active())),
            )

            for job in job_queue.job_list:
                job.wait_for()

            assert job_queue.fetch_next_waiting() is None

            for job in job_queue.job_list:
                assert job.status == JobStatusType.JOB_QUEUE_FAILED
            
            assert True
