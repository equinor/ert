#!/usr/bin/env python

import sys
import os
import os.path
from datetime import datetime as dt
import time
import signal
import random
import shutil



from res.job_queue import JobManager, assert_file_executable

REQUESTED_HEXVERSION  =  0x02070000

LOG_URL       = "http://devnull.statoil.no:4444"

SHORT_SLEEP = 1
LONG_SLEEP = 5
EXIT_file = 'ERROR'

def check_version():
    if sys.hexversion < REQUESTED_HEXVERSION:
        version = sys.version_info
        warning = """
/------------------------------------------------------------------------
| You are running Python version %d.%d.%d; much of the ert functionality
| expects to be running on Python 2.7.5.  Version 2.7.13 is the default
| version in /prog/sdpsoft.
|
| It is highly recommended that you update your setup to use Python 2.7.5.
|
\\------------------------------------------------------------------------

""" % (version[0] , version[1] , version[2])
        sys.stderr.write( warning )





def redirect(fname , fd , open_mode):
    new_fd = os.open(fname , open_mode )
    os.dup2(new_fd , fd)
    os.close(new_fd)

def redirect_input(fname , fd ):
    redirect( fname , fd , os.O_RDONLY)


def redirect_output(fname , fd ,start_time):
    if os.path.isfile(fname):
        mtime = os.path.getmtime( fname )
        if mtime < start_time:
            # Old stale version; truncate.
            redirect( fname , fd , os.O_WRONLY | os.O_TRUNC | os.O_CREAT )
        else:
            # A new invocation of the same job instance; append putput
            redirect( fname , fd , os.O_APPEND )
    else:
        redirect( fname , fd , os.O_WRONLY | os.O_TRUNC | os.O_CREAT )



def cond_symlink(target , src):
    if not os.path.exists(src):
        os.symlink( target , src )


def cond_unlink(fname):
    if os.path.exists(fname):
        os.unlink(fname)



def exec_job(job , executable, start_time):
    assert_file_executable(executable)

    if job.get("stdin"):
        redirect_input(job["stdin"]  , 0)

    if job.get("stdout"):
        redirect_output(job["stdout"] , 1 , start_time )

    if job.get("stderr"):
        redirect_output(job["stderr"] , 2 , start_time )

    if job.get("environment"):
        env = job["environment"]
        for key in env.keys():
            os.putenv(key , env[key])
    argList = [ executable ]
    if job.get("argList"):
        argList += job["argList"]
    os.execvp(executable , argList )




def kill_job_and_EXIT( job , P ):
    dump_EXIT_file( job , "Job:%s has been running for more than %d minutes - explicitly killed.\n" % (job["name"] , job["max_running_minutes"]))
    pgid = os.getpgid( P.pid )
    os.killpg(pgid , signal.SIGKILL )

    # The os.killpg() will kill this script as well; including the sys.exit() to extra certain.
    sys.exit( 1 )


def unlink_empty(fname):
    if os.path.exists(fname):
        st = os.stat( fname )
        if st.st_size == 0:
            os.unlink( fname )



def cleanup( job ):
    if job.get("stdout"):
        unlink_empty( job["stdout"] )
    if job.get("stderr"):
        unlink_empty( job["stderr"] )
    if job.get("license_link"):
        os.unlink(job["license_link"])



# This function implements a simple "personal license" system limiting
# how many instances of this job can run concurrently. Observe that the
# limiting is based on pr. invocation of the queue system (i.e. ERT
# binary) and pr user. The system works as follows:
#
#  1. The job is initilized with a license_path and a max_running
#     variable.
#
#  2. In the licens_path directory a license file is made.
#
#  3. For each instance a random hard-link is created to the license
#     file - this is how the number of concurrent uses is counted.
#
#  4. When the external program is finished the hard link is removed.


def license_check( job ):
    job["license_link"] = None
    if "max_running" in job:
        if job["max_running"]:
            job["license_file"] = "%s/%s" % (job["license_path"] , job["name"])
            max_running         = job["max_running"]
            license_file        = job["license_file"]
            while True:
                job["license_link"] = "%s/%d" % (job["license_path"] , random.randint(100000,999999))
                if not os.path.exists(job["license_link"]):
                    break


            if not os.path.exists(license_file):
                fileH = open(license_file , "w")
                fileH.write("This is a license file for job:%s" % job["name"])
                fileH.close()


            stat_info = os.stat(license_file)
            currently_running = stat_info[3] - 1

            while True:
                stat_info = os.stat(license_file)
                currently_running = stat_info[3] - 1
                if currently_running < max_running:
                    break
                else:
                    time.sleep(LONG_SLEEP)

            os.link(license_file , job["license_link"])

            while True:
                stat_info = os.stat(license_file)
                currently_running = stat_info[3] - 1
                if currently_running <= max_running:
                    break # OK - now we can leave the building - and let the job start
                else:
                    time.sleep(LONG_SLEEP)


# This file will be read by the job_queue_node_fscanf_EXIT() function
# in job_queue.c. Be very careful with changes in output format.
def dump_EXIT_file( job , error_msg):
    fileH = open(EXIT_file , "a")
    now = dt.now()
    fileH.write("<error>\n")
    fileH.write("  <time>%02d:%02d:%02d</time>\n" % (now.tm_hour , now.tm_min , now.tm_sec))
    fileH.write("  <job>%s</job>\n" % job["name"])
    fileH.write("  <reason>%s</reason>\n" % error_msg)
    stderr_file = None
    if job["stderr"]:
        if os.path.exists( job["stderr"]):
            errH = open( job["stderr"] , "r")
            stderr = errH.read()
            if stderr:
                stderr_file = os.path.join( os.getcwd(), job["stderr"] )
            else:
                stderr = "<Not written by:%s>\n" % job["name"]
            errH.close()
        else:
            stderr = "<stderr: Could not find file:%s>\n" % job["stderr"]
    else:
        stderr = "<stderr: Not redirected>\n"

    fileH.write("  <stderr>\n%s</stderr>\n" % stderr)
    if stderr_file:
        fileH.write("  <stderr_file>%s</stderr_file>\n" % stderr_file)

    fileH.write("</error>\n")
    fileH.close()

    # Have renamed the exit file from "EXIT" to "ERROR";
    # must keep the old "EXIT" file around until all old ert versions
    # are flushed out.
    shutil.copyfile( EXIT_file , "EXIT")


def waitForTargetFile(target_file , stat_start_time):
    timeout = 60
    start_time = time.time()
    while True:
        if os.path.exists(target_file):
            stat = os.stat(target_file)
            if stat.st_mtime > stat_start_time:
                return (True , "")

        time.sleep(SHORT_SLEEP)
        if time.time() - start_time > timeout:
            break

    # We have gone out of the loop via the break statement,
    # i.e. on a timeout.
    if os.path.exists(target_file):
        stat = os.stat(target_file)
        return (False , "The target file:%s has not been updated; this is flagged as failure. mtime:%s   stat_start_time:%s" % (target_file , stat.st_mtime , stat_start_time))
    else:
        return (False , "Could not find target_file:%s" % target_file)



def run_one(job_manager , job):
    license_check( job )
    if job.get("stdin"):
        if not os.path.exists(job["stdin"]):
            return (False , 0 , "Could not locate stdin file: %s" % job["stdin"])

    if job.get("start_file"):
        if not os.path.exists(job["start_file"]):
            return (False , -1 , "Could not locate start_file:%s" % job["start_file"])

    if job.get("error_file"):
        if os.path.exists( job.get("error_file")):
            os.unlink( job.get("error_file") )

    stat_file = None
    start_time = time.time()
    if job.get("target_file"):
        if os.path.exists(job["target_file"]):
            stat = os.stat(job["target_file"])
            stat_start_time = stat.st_mtime
        else:
            stat_file = "%s-stat-target" % job.get("target_file")

            f = open(stat_file , "w")
            f.write("This file is here only as a stat() reference")
            f.close()

            stat = os.stat(stat_file)
            stat_start_time  = stat.st_mtime - 1


    if job.get("max_running_minutes"):
        P = job_manager.jobProcess(job)
        while True:
            time.sleep( SHORT_SLEEP )
            run_time = time.time() - start_time
            returncode = P.poll()
            if returncode is None:
                # Still running
                if run_time > job["max_running_minutes"] * 60:
                    # Have been running to long - kill it
                    kill_job_and_EXIT( job , P )
            else:
                # Have completed within the time limits
                break

        if returncode == 0:
            exit_status = 0
        else:
            exit_status = 1

    else:
        exit_status, error_msg = job_manager.runJob(job)


    # Check sucess of job; look for both target_file and
    # error_file. Both can be used to signal failure
    # independently.

    status = None

    # The target_file can be used *both* to set status OK and to set status failed.
    if exit_status == 0:
        if job.get("target_file"):
            (ok , msg) = waitForTargetFile( job.get("target_file") , stat_start_time )
            status = (ok , exit_status , msg)


    # The error_file can only be used to set status failed.
    if exit_status == 0:
        if job.get("error_file"):
            if os.path.exists( job.get("error_file") ):
                status = (False , 1 , "Found the error file:%s - job failed." % job.get("error_file"))

    # Neither error_file nor target_file have been specified; then we
    # use the OS exit status to check.
    if status is None:
        if exit_status == 0:
            status = (True , 0 , "")
        else:
            status = (False, exit_status , error_msg)

    return status


def main(argv):
    """FIXME the argument argv is ignored."""

    if len(sys.argv) >= 2:
        run_path =  sys.argv[1]

        if not os.path.exists( run_path ):
            sys.stderr.write("*****************************************************************\n")
            sys.stderr.write("** FATAL Error: Could not find directory: %s \n" % run_path)
            sys.stderr.write("** CWD: %s\n" % os.getcwd())
            sys.stderr.write("*****************************************************************\n")

            sys.exit(-1)
        os.chdir( run_path )


    #################################################################
    # 1. Modify the sys.path variable to include the runpath
    # 2. Import the jobs module.
    #################################################################
    random.seed()
    check_version()

    job_manager = JobManager(error_url=LOG_URL, log_url=LOG_URL)

    if len(sys.argv) <= 2:
        # Normal batch run.

        # Set this to true to ensure that empty job lists come out successfully.
        OK = True

        for job in job_manager:
            job_manager.startStatus( job )
            (OK , exit_status, error_msg) = run_one( job_manager , job)
            job_manager.completeStatus(exit_status, error_msg, job=job)
            if not OK:
                job_manager.exit( job, exit_status , error_msg )

        job_manager.complete()
        if OK:
            job_manager.createOKFile( )

    else:
        #Interactive run

        for job_name in sys.argv[2:]:
            # This is totally unpredictable if there more jobs with
            # the same name.
            if job_name in job_manager:
                job = job_manager[job_name]
                print("Running job: %s ... " % job_name)
                sys.stdout.flush()
                (OK , exit_status, error_msg) = run_one( job_manager, job )
                if OK:
                    print("OK")
                else:
                    print("failed ....")
                    print("-----------------------------------------------------------------")
                    if job.get("stderr"):
                        print("Error:%s " % error_msg)
                        if os.path.exists(job["stderr"]):
                            fileH = open(job["stderr"],"r")
                            for line in fileH.readlines():
                                print(line),
                            fileH.close()
                    print("-----------------------------------------------------------------")
                    sys.exit()
            else:
                print("Job: %s does not exist. Available jobs:" % job_name)
                for j in job_manager.job_list:
                    print("   %s" % j["name"])


#################################################################

#################################################################
#os.setsid( )
os.nice(19)
if __name__ == "__main__":
    main( sys.argv )
