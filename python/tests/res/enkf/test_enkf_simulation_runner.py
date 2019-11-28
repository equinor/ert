from res.enkf import EnkfSimulationRunner
from res.job_queue import JobQueueNode, JobStatusType, JobQueue, JobQueueManager
from res.job_queue import ThreadStatus, Driver, QueueDriverEnum
from tests.utils import tmpdir
from unittest import TestCase
import os, stat, random


class MockedQueue:
    def __init__(self, job_list):
        self.job_list = job_list

    def count_status(self, status):
        return len([x for x in self.job_list if x.status == status])


class MockedJob:
    def __init__(self, status):
        self.status = status
        self._start_time = 0
        self._current_time = 0

    @property
    def runtime(self):
        return self._end_time - self._start_time

    def stop(self):
        self.status = JobStatusType.JOB_QUEUE_FAILED


class EnkfSimulationRunnerTest(TestCase):
    def test_stop_long_running(self):
        """
        This test should verify that only the jobs that are has a runtime
        25% longer than the average completed are stopped when
        stop_long_running_jobs is called.
        """
        job_list = [MockedJob(JobStatusType.JOB_QUEUE_WAITING) for i in range(10)]

        for i in range(5):
            job_list[i].status = JobStatusType.JOB_QUEUE_DONE
            job_list[i]._start_time = 0
            job_list[i]._end_time = 10

        for i in range(5, 8):
            job_list[i].status = JobStatusType.JOB_QUEUE_RUNNING
            job_list[i]._start_time = 0
            job_list[i]._end_time = 20

        for i in range(8, 10):
            job_list[i].status = JobStatusType.JOB_QUEUE_RUNNING
            job_list[i]._start_time = 0
            job_list[i]._end_time = 5

        queue = MockedQueue(job_list)

        EnkfSimulationRunner.stop_long_running_jobs(queue, 5)

        for i in range(5):
            assert job_list[i].status == JobStatusType.JOB_QUEUE_DONE

        for i in range(5, 8):
            assert job_list[i].status == JobStatusType.JOB_QUEUE_FAILED

        for i in range(8, 10):
            assert job_list[i].status == JobStatusType.JOB_QUEUE_RUNNING
