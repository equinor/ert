from res.job_queue import JobStatusType, Driver, QueueDriverEnum, JobQueue, JobQueueNode
from tests import ResTest
from ecl.util.test import TestAreaContext
import os, stat, time

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

simple_script = "#!/usr/bin/env python\n"\
                        "print('hello')\n"\
                        "\n"

never_ending_script = "#!/usr/bin/env python\n"\
                        "while(true):\n"\
                        "   pass"\
                        "\n"

failing_script = "#!/usr/bin/env python\n"\
                        "import sys\n"\
                        "sys.exit(1)"\
                        "\n"


def create_queue(script):
    driver = Driver(driver_type=QueueDriverEnum.LOCAL_DRIVER, max_running=5)
    job_queue = JobQueue(driver)
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

def start_all(job_queue):
    job = job_queue.fetch_next_waiting()
    threads = []
    while(job is not None):
        threads.append(job.run(job_queue.driver))
        job = job_queue.fetch_next_waiting()
    return threads

class JobQueueTest(ResTest):

    def testStatusEnum(self):
        source_path = "lib/include/ert/job_queue/job_status.hpp"
        self.assertEnumIsFullyDefined(JobStatusType, "job_status_type", source_path)

    def test_kill_jobs(self):
        with TestAreaContext("job_queue_test_add") as work_area:
            job_queue = create_queue(never_ending_script)

            assert job_queue.queue_size == 10
            assert job_queue.is_running()
            
            threads = start_all(job_queue)
         
            job_queue.kill_all_jobs()
            assert not job_queue.is_running()
            
            for job in job_queue.job_list:
                assert job.status == JobStatusType.JOB_QUEUE_IS_KILLED
            
            for t in threads:
                t.join()
            assert True

    def test_add_jobs(self):
        with TestAreaContext("job_queue_test_add") as work_area:
            job_queue = create_queue(simple_script)

            assert job_queue.queue_size == 10
            assert job_queue.is_running()
            assert job_queue.fetch_next_waiting() is not None
            
            job_queue.kill_all_jobs()
            assert not job_queue.is_running()


    def test_failing_jobs(self):
        with TestAreaContext("job_queue_test_add") as work_area:
            job_queue = create_queue(failing_script)

            assert job_queue.queue_size == 10
            assert job_queue.is_running()
            
            threads = start_all(job_queue)
            assert job_queue.fetch_next_waiting() is None
            time.sleep(2)
            assert not job_queue.is_running()

            for job in job_queue.job_list:
                assert job.status == JobStatusType.JOB_QUEUE_EXIT
            
            for t in threads:
                t.join()
            assert True