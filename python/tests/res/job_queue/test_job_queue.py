from res.job_queue import JobStatusType
from tests import ResTest


class JobQueueTest(ResTest):

    def testStatusEnum(self):
        source_path = "lib/include/ert/job_queue/job_status.hpp"
        self.assertEnumIsFullyDefined(JobStatusType, "job_status_type", source_path)


