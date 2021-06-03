from res.job_queue import JobStatusType, JobQueue
from unittest import TestCase, mock


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

    def convertToCReference(self, _):
        pass


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

        # The driver is of no consequence, so resolving it in the c layer is
        # uninteresting and mocked out.
        with mock.patch("res.job_queue.JobQueue._set_driver"):
            queue = JobQueue(mock.MagicMock())

            # We don't need the c layer call here, we only need it added to
            # the queue's job_list.
            with mock.patch("res.job_queue.JobQueue._add_job") as _add_job:
                for idx, job in enumerate(job_list):
                    _add_job.return_value = idx
                    queue.add_job(job, idx)

        queue.stop_long_running_jobs(5)
        queue._transition()

        for i in range(5):
            assert job_list[i].status == JobStatusType.JOB_QUEUE_DONE
            assert queue.snapshot()[i] == str(JobStatusType.JOB_QUEUE_DONE)

        for i in range(5, 8):
            assert job_list[i].status == JobStatusType.JOB_QUEUE_FAILED
            assert queue.snapshot()[i] == str(JobStatusType.JOB_QUEUE_FAILED)

        for i in range(8, 10):
            assert job_list[i].status == JobStatusType.JOB_QUEUE_RUNNING
            assert queue.snapshot()[i] == str(JobStatusType.JOB_QUEUE_RUNNING)
