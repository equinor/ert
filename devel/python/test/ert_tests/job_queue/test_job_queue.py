from ert.job_queue import JobStatusType
from ert_tests import ExtendedTestCase


class JobQueueTest(ExtendedTestCase):

    def testStatusEnum(self):
        self.assertEqual(JobStatusType.JOB_QUEUE_NOT_ACTIVE, 1)
        # self.assertEqual(JobStatusType.JOB_QUEUE_LOADING, 2)
        self.assertEqual(JobStatusType.JOB_QUEUE_WAITING, 4)
        self.assertEqual(JobStatusType.JOB_QUEUE_SUBMITTED, 8)
        self.assertEqual(JobStatusType.JOB_QUEUE_PENDING, 16)
        self.assertEqual(JobStatusType.JOB_QUEUE_RUNNING, 32)
        self.assertEqual(JobStatusType.JOB_QUEUE_DONE, 64)
        self.assertEqual(JobStatusType.JOB_QUEUE_EXIT, 128)
        # self.assertEqual(JobStatusType.JOB_QUEUE_RUN_OK, 256)
        # self.assertEqual(JobStatusType.JOB_QUEUE_RUN_FAIL, 512)
        # self.assertEqual(JobStatusType.JOB_QUEUE_ALL_OK, 1024)
        # self.assertEqual(JobStatusType.JOB_QUEUE_ALL_FAIL, 2048)
        self.assertEqual(JobStatusType.JOB_QUEUE_USER_KILLED, 4096)
        self.assertEqual(JobStatusType.JOB_QUEUE_USER_EXIT, 8192)
        self.assertEqual(JobStatusType.JOB_QUEUE_SUCCESS, 16384)
        self.assertEqual(JobStatusType.JOB_QUEUE_RUNNING_CALLBACK, 32768)
        self.assertEqual(JobStatusType.JOB_QUEUE_FAILED, 65536)