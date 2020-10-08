#  Copyright (C) 2013  Equinor ASA, Norway.
#
#  The file 'job_status_type_enum.py' is part of ERT - Ensemble based Reservoir Tool.
#
#  ERT is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
#  for more details.
from cwrap import BaseCEnum


class JobStatusType(BaseCEnum):
    TYPE_NAME = "job_status_type_enum"
    JOB_QUEUE_NOT_ACTIVE = None  # This value is used in external query routines - for jobs which are (currently) not active. */
    JOB_QUEUE_WAITING = None  # A node which is waiting in the internal queue.
    JOB_QUEUE_SUBMITTED = None  # Internal status: It has has been submitted - the next status update will (should) place it as pending or running.
    JOB_QUEUE_PENDING = None  # A node which is pending - a status returned by the external system. I.e LSF
    JOB_QUEUE_RUNNING = None  # The job is running
    JOB_QUEUE_DONE = None  # The job is done - but we have not yet checked if the target file is produced */
    JOB_QUEUE_EXIT = None  # The job has exited - check attempts to determine if we retry or go to complete_fail   */
    JOB_QUEUE_IS_KILLED = None  # The job has been killed, following a  JOB_QUEUE_DO_KILL - can restart. */
    JOB_QUEUE_DO_KILL = None  # The the job should be killed, either due to user request, or automated measures - the job can NOT be restarted.. */
    JOB_QUEUE_SUCCESS = None
    JOB_QUEUE_RUNNING_DONE_CALLBACK = None
    JOB_QUEUE_RUNNING_EXIT_CALLBACK = None
    JOB_QUEUE_STATUS_FAILURE = None
    JOB_QUEUE_FAILED = None
    JOB_QUEUE_DO_KILL_NODE_FAILURE = None
    JOB_QUEUE_UNKNOWN = None

    @classmethod
    def from_string(cls, string):
        return super().from_string(string)


JobStatusType.addEnum("JOB_QUEUE_NOT_ACTIVE", 1)
JobStatusType.addEnum("JOB_QUEUE_WAITING", 2)
JobStatusType.addEnum("JOB_QUEUE_SUBMITTED", 4)
JobStatusType.addEnum("JOB_QUEUE_PENDING", 8)
JobStatusType.addEnum("JOB_QUEUE_RUNNING", 16)
JobStatusType.addEnum("JOB_QUEUE_DONE", 32)
JobStatusType.addEnum("JOB_QUEUE_EXIT", 64)
JobStatusType.addEnum("JOB_QUEUE_IS_KILLED", 128)
JobStatusType.addEnum("JOB_QUEUE_DO_KILL", 256)
JobStatusType.addEnum("JOB_QUEUE_SUCCESS", 512)
JobStatusType.addEnum("JOB_QUEUE_RUNNING_DONE_CALLBACK", 1024)
JobStatusType.addEnum("JOB_QUEUE_RUNNING_EXIT_CALLBACK", 2048)
JobStatusType.addEnum("JOB_QUEUE_STATUS_FAILURE", 4096)
JobStatusType.addEnum("JOB_QUEUE_FAILED", 8192)
JobStatusType.addEnum("JOB_QUEUE_DO_KILL_NODE_FAILURE", 16384)
JobStatusType.addEnum("JOB_QUEUE_UNKNOWN", 32768)
