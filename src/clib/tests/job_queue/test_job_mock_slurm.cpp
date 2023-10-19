#include "../tmpdir.hpp"
#include "catch2/catch.hpp"
#include <filesystem>
#include <sys/stat.h>
#include <vector>

#include <ert/job_queue/queue_driver.hpp>
#include <ert/job_queue/slurm_driver.hpp>

/*
  This test tests the interaction between slurm and libres by using mock scripts
  for slurm functionality. The scripts are very simple, and just return suitable
  "results" on stdout which the slurm driver in libres interprets. Observe that
  this testing is 100% stateless and does not invoke running external "jobs" in
  any kind.

  The different jobs have different behaviour based on the job name; the
  names are just the strings "0", "1", "2", "3" and "4". The behaviour of the
  different job is as follows:

  0: The sbatch command fails with exit != 0 when this job is started. The test
     verifies that the queue_driver_submit_job() returns nullptr, and after that
     we do not hear anything more from this job.

  1: This job is submitted as it should, but then subsequently cancelled. The
     cancel script actually does not do anything, but afterward we test that
     the status is correctly returned. Before the CANCELLED status is reached we
     go through two steps:

     a) We run the squeue command, that does not report on cancelled jobs and
        will not report status for job 1.

     b) We run the scontrol command which through a detailed request finds the
        cancelled status of the job.

  2: / 3: These jobs are PENDING and RUNNING respectively, that status is
     reported by the squeue command.

  4: This job has been completed. As with the canceled job 1 we need to go
     through both squeue and scontrol before we get the status of the job.

*/

void make_sleep_job(const char *fname, int sleep_time) {
    FILE *stream = fopen(fname, "w");
    REQUIRE(stream != nullptr);
    fprintf(stream, "sleep %d \n", sleep_time);
    fclose(stream);
    chmod(fname, S_IRWXU);
}

void make_script(const char *fname, const std::string &content) {
    FILE *stream = fopen(fname, "w");
    REQUIRE(stream != nullptr);
    fprintf(stream, "%s", content.c_str());
    fclose(stream);
    chmod(fname, S_IRWXU);
}

void install_script(queue_driver_type *driver, const char *option,
                    const std::string &content) {
    std::string fname =
        std::filesystem::current_path().string() + "/" + std::string(option);
    make_script(fname.c_str(), content);
    queue_driver_set_option(driver, option, fname.c_str());
}

void make_slurm_commands(queue_driver_type *driver) {
    std::string sbatch = R"(#!/usr/bin/env bash

if [ $2 = "--job-name=0" ]; then
   exit 1
fi

if [ $2 = "--job-name=1" ]; then
   echo 1
fi

if [ $2 = "--job-name=2" ]; then
   echo 2
fi

if [ $2 = "--job-name=3" ]; then
   echo 3
fi

if [ $2 = "--job-name=4" ]; then
   echo 4
fi
)";

    std::string scancel = R"(#!/usr/bin/env bash

exit 0
)";

    std::string scontrol = R"(#!/usr/bin/env bash
if [ $3 = "1" ]; then
cat <<EOF
   UserId=user(777) GroupId=group(888) MCS_label=N/A
   Priority=4294901494 Nice=0 Account=(null) QOS=(null)
   JobState=CANCELLED Reason=None Dependency=(null)
   Requeue=1 Restarts=0 BatchFlag=1 Reboot=0 ExitCode=0:0
   RunTime=00:00:22 TimeLimit=UNLIMITED TimeMin=N/A
   SubmitTime=2020-06-08T19:43:54 EligibleTime=2020-06-08T19:43:54
   AccrueTime=Unknown
   StartTime=2020-06-08T19:43:55 EndTime=Unknown Deadline=N/A
   PreemptTime=None SuspendTime=None SecsPreSuspend=0
   LastSchedEval=2020-06-08T19:43:55
   Partition=debug AllocNode:Sid=ws:13314
   ReqNodeList=(null) ExcNodeList=(null)
   NodeList=ws
   BatchHost=ws
   NumNodes=1 NumCPUs=1 NumTasks=1 CPUs/Task=1 ReqB:S:C:T=0:0:*:*
   TRES=cpu=1,node=1,billing=1
   Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=*
   MinCPUsNode=1 MinMemoryNode=0 MinTmpDiskNode=0
   Features=(null) DelayBoot=00:00:00
   OverSubscribe=OK Contiguous=0 Licenses=(null) Network=(null)
   Command=/tmp/flow.sh
   WorkDir=/home/user/work/ERT/libres/build
   StdErr=/home/user/work/ERT/libres/build/slurm-7625.out
   StdIn=/dev/null
   StdOut=/home/user/work/ERT/libres/build/slurm-7625.out
   Power=

EOF
fi

if [ $3 = "4" ]; then
cat <<EOF
   UserId=user(777) GroupId=group(888) MCS_label=N/A
   Priority=4294901494 Nice=0 Account=(null) QOS=(null)
   JobState=COMPLETED Reason=None Dependency=(null)
   Requeue=1 Restarts=0 BatchFlag=1 Reboot=0 ExitCode=0:0
   RunTime=00:00:22 TimeLimit=UNLIMITED TimeMin=N/A
   SubmitTime=2020-06-08T19:43:54 EligibleTime=2020-06-08T19:43:54
   AccrueTime=Unknown
   StartTime=2020-06-08T19:43:55 EndTime=Unknown Deadline=N/A
   PreemptTime=None SuspendTime=None SecsPreSuspend=0
   LastSchedEval=2020-06-08T19:43:55
   Partition=debug AllocNode:Sid=ws:13314
   ReqNodeList=(null) ExcNodeList=(null)
   NodeList=ws
   BatchHost=ws
   NumNodes=1 NumCPUs=1 NumTasks=1 CPUs/Task=1 ReqB:S:C:T=0:0:*:*
   TRES=cpu=1,node=1,billing=1
   Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=*
   MinCPUsNode=1 MinMemoryNode=0 MinTmpDiskNode=0
   Features=(null) DelayBoot=00:00:00
   OverSubscribe=OK Contiguous=0 Licenses=(null) Network=(null)
   Command=/tmp/flow.sh
   WorkDir=/home/user/work/ERT/libres/build
   StdErr=/home/user/work/ERT/libres/build/slurm-7625.out
   StdIn=/dev/null
   StdOut=/home/user/work/ERT/libres/build/slurm-7625.out
   Power=

EOF
fi

)";

    std::string squeue = R"(#!/usr/bin/env bash
echo "2 PENDING"
echo "3 RUNNING"
)";

    install_script(driver, SLURM_SBATCH_OPTION, sbatch);
    install_script(driver, SLURM_SCANCEL_OPTION, scancel);
    install_script(driver, SLURM_SCONTROL_OPTION, scontrol);
    install_script(driver, SLURM_SQUEUE_OPTION, squeue);
}

void *submit_job(queue_driver_type *driver, std::string cwd,
                 const std::string &job_name, const char *cmd) {
    std::string run_path = cwd + "/" + job_name;
    std::filesystem::create_directory(std::filesystem::path(run_path));
    return queue_driver_submit_job(driver, cmd, 1, run_path.c_str(),
                                   job_name.c_str());
}

TEST_CASE("job_mock_slurm_run", "[job_slurm]") {
    TmpDir tmpdir; // cwd is now a generated tmpdir
    queue_driver_type *driver = queue_driver_alloc(SLURM_DRIVER);

    std::string cwd = tmpdir.get_current_tmpdir();
    std::string cmd_path = cwd + "/cmd.sh";
    const char *cmd = cmd_path.c_str();

    std::vector<void *> jobs;
    make_sleep_job(cmd, 10);
    make_slurm_commands(driver);

    REQUIRE(submit_job(driver, cwd, "0", cmd) == nullptr);

    auto job0 = submit_job(driver, cwd, "1", cmd);
    REQUIRE_FALSE(job0 == nullptr);
    jobs.push_back(job0);

    auto job1 = submit_job(driver, cwd, "2", cmd);
    REQUIRE_FALSE(job1 == nullptr);
    jobs.push_back(job1);

    auto job2 = submit_job(driver, cwd, "3", cmd);
    REQUIRE_FALSE(job2 == nullptr);
    jobs.push_back(job2);

    auto job3 = submit_job(driver, cwd, "4", cmd);
    REQUIRE_FALSE(job3 == nullptr);
    jobs.push_back(job3);

    queue_driver_kill_job(driver, jobs[0]);
    REQUIRE(queue_driver_get_status(driver, jobs[0]) == JOB_QUEUE_IS_KILLED);
    REQUIRE(queue_driver_get_status(driver, jobs[1]) == JOB_QUEUE_PENDING);
    REQUIRE(queue_driver_get_status(driver, jobs[2]) == JOB_QUEUE_RUNNING);
    REQUIRE(queue_driver_get_status(driver, jobs[3]) == JOB_QUEUE_DONE);

    for (auto job : jobs)
        queue_driver_free_job(driver, job);

    queue_driver_free(driver);
}
