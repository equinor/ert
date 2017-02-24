import os.path
import json

from ert.test import TestAreaContext, ExtendedTestCase
from ert.util import SubstitutionList
from ert.job_queue.forward_model import ForwardModel
from ert.job_queue.ext_job import ExtJob
from ert.job_queue.ext_joblist import ExtJoblist

#
# Data for testing
#
joblist = [
            {
                "name" : "PERLIN",
                "executable" : "perlin.py",
                "target_file" : "my_target_file",
                "error_file" : "error_file",
                "start_file" : "some_start_file",
                "stdout" : "perlin.stdout",
                "stderr" : "perlin.stderr",
                "stdin" : "input4thewin",
                "argList" : ["-speed", "hyper"],
                "environment" : {"TARGET": "flatland"},
                "license_path" : "this/is/my/license/PERLIN",
                "max_running_minutes" : 12,
                "max_running" : 30
            },
            {
                "name" : "AGGREGATOR",
                "executable" : "aggregator.py",
                "target_file" : "target",
                "error_file" : "None",
                "start_file" : "eple",
                "stdout" : "aggregator.stdout",
                "stderr" : "aggregator.stderr",
                "stdin" : "illgiveyousome",
                "argList" : ["-o"],
                "environment" : {"STATE": "awesome"},
                "license_path" : "I/will/pay/ya/tomorrow/AGGREGATOR",
                "max_running_minutes" : 1,
                "max_running" : 14,
            },
            {
                "name" : "PI",
                "executable" : "pi.py",
                "target_file" : "my_target_file",
                "error_file" : "error_file",
                "start_file" : "some_start_file",
                "stdout" : "pi.stdout",
                "stderr" : "pi.stderr",
                "stdin" : "input4thewin",
                "argList" : ["-p", "8"],
                "environment" : {"LOCATION": "earth"},
                "license_path" : "license/PI",
                "max_running_minutes" : 12,
                "max_running" : 30
            },
            {
                "name" : "OPTIMUS",
                "executable" : "optimus.py",
                "target_file" : "target",
                "error_file" : "None",
                "start_file" : "eple",
                "stdout" : "optimus.stdout",
                "stderr" : "optimus.stderr",
                "stdin" : "illgiveyousome",
                "argList" : ["-help"],
                "environment" : {"PATH": "/ubertools/4.1"},
                "license_path" : "license/OPTIMUS",
                "max_running_minutes" : 1,
                "max_running" : 14,
            }
        ]

#
# Keywords for the ext_job initialization file
#
ext_job_keywords = [
            "MAX_RUNNING",
            "STDIN",
            "STDOUT",
            "STDERR",
            "EXECUTABLE",
            "TARGET_FILE",
            "ERROR_FILE",
            "START_FILE",
            "ARGLIST",
            "ENV",
            "MAX_RUNNING_MINUTES"
            ]

#
# JSON keywords
#
json_keywords = [
            "name",
            "executable",
            "target_file",
            "error_file",
            "start_file",
            "stdout",
            "stderr",
            "stdin",
            "license_path",
            "max_running_minutes",
            "max_running",
            "argList",
            "environment"
            ]

def str_none_sensitive(x):
    return str(x) if x is not None else None

DEFAULT_NAME = "default_job_name"
def _generate_job(
                  name,
                  executable,
                  target_file,
                  error_file,
                  start_file,
                  stdout,
                  stderr,
                  stdin,
                  environment,
                  arglist,
                  max_running_minutes,
                  max_running,
                  license_root_path,
                  private
                ):
    config_file = (DEFAULT_NAME if name is None else name)
    environment = (
            None if environment is None else
            environment.keys() + environment.values()
            )

    values = [  str_none_sensitive(max_running),
                stdin,
                stdout,
                stderr,
                executable,
                target_file,
                error_file,
                start_file,
                None if arglist is None else " ".join(arglist),
                None if environment is None else " ".join(environment),
                str_none_sensitive(max_running_minutes)
                ]

    conf = open(config_file, "w")
    for key, val in zip(ext_job_keywords, values):
        if val is not None:
            conf.write("%s %s\n" %(key, val))
    conf.close()

    exec_file = open(executable, "w")
    exec_file.close()

    ext_job = ExtJob(config_file, private, name, license_root_path)
    os.unlink(config_file)
    os.unlink(executable)

    return ext_job

def empty_list_if_none(l):
        return [] if l is None else l

def default_name_if_none(name):
    return DEFAULT_NAME if name is None else name

def get_license_root_path(license_path):
    return os.path.split(license_path)[0]

def dump_config_to_terminal():
    print "############ JSON ####################"
    with open("jobs.json", "r") as f:
        print f.read()

    print "############ PY ######################"
    with open("jobs.py", "r") as f:
        print f.read()

    print "######################################"
 
def load_configs(config_file):
    with open(config_file, "r") as cf:
        jobs = json.load(cf)

    return jobs

class ForwardModelFormattedPrintTest(ExtendedTestCase):

    JOBS_JSON_FILE = "jobs.json"

    @classmethod
    def setUpClass(cls):
        # Make all executable paths absolute
        for job in joblist:
            if not os.path.isabs(job["executable"]):
                job["executable"] = os.path.join(
                        os.getcwd(),
                        job["executable"]
                        )

    def validate_ext_job(self, ext_job, ext_job_config):
        zero_if_none = lambda x: 0 if x is None else x

        self.assertEqual(
                ext_job.name(),
                default_name_if_none(ext_job_config["name"])
                )
        self.assertEqual(
                ext_job.get_executable(),
                ext_job_config["executable"]
                )
        self.assertEqual(
                ext_job.get_target_file(),
                ext_job_config["target_file"]
                )
        self.assertEqual(
                ext_job.get_error_file(),
                ext_job_config["error_file"]
                )
        self.assertEqual(
                ext_job.get_start_file(),
                ext_job_config["start_file"]
                )
        self.assertEqual(
                ext_job.get_stdout_file(),
                ext_job_config["stdout"]
                )
        self.assertEqual(
                ext_job.get_stderr_file(),
                ext_job_config["stderr"]
                )
        self.assertEqual(
                ext_job.get_stdin_file(),
                ext_job_config["stdin"]
                )
        self.assertEqual(
                ext_job.get_max_running_minutes(),
                zero_if_none(ext_job_config["max_running_minutes"])
                )
        self.assertEqual(
                ext_job.get_max_running(),
                zero_if_none(ext_job_config["max_running"])
                )
        self.assertEqual(
                ext_job.get_license_path(),
                ext_job_config["license_path"]
                )
        self.assertEqual(
                ext_job.get_arglist(),
                empty_list_if_none(ext_job_config["argList"])
                )

        if ext_job_config["environment"] is None:
            self.assertTrue(len(ext_job.get_environment().keys()) == 0)
        else:
            self.assertEqual(
                    ext_job.get_environment().keys(),
                    ext_job_config["environment"].keys()
                    )
            for key in ext_job_config["environment"].keys():
                self.assertEqual(
                        ext_job.get_environment()[key],
                        ext_job_config["environment"][key]
                        )

    def generate_job_from_dict(self, ext_job_config, private = True):
        ext_job = _generate_job(
            ext_job_config["name"],
            ext_job_config["executable"],
            ext_job_config["target_file"],
            ext_job_config["error_file"],
            ext_job_config["start_file"],
            ext_job_config["stdout"],
            ext_job_config["stderr"],
            ext_job_config["stdin"],
            ext_job_config["environment"],
            ext_job_config["argList"],
            ext_job_config["max_running_minutes"],
            ext_job_config["max_running"],
            get_license_root_path(ext_job_config["license_path"]),
            private
            );
        
        self.validate_ext_job(ext_job, ext_job_config)
        return ext_job

    def set_up_forward_model(self, selected_jobs=range(len(joblist))):
        jobs = [self.generate_job_from_dict(job) for job in joblist]

        ext_joblist = ExtJoblist()
        for job in jobs:
            ext_joblist.add_job(job.name(), job)

        forward_model = ForwardModel(ext_joblist)
        for index in selected_jobs:
            forward_model.add_job(jobs[index].name())

        return forward_model

    def verify_json_dump(self, selected_jobs, global_args, umask):
        self.assertTrue(os.path.isfile(self.JOBS_JSON_FILE))
        config = load_configs(self.JOBS_JSON_FILE)

        self.assertEqual(umask, int(config["umask"], 8))
        self.assertEqual(len(selected_jobs), len(config["jobList"]))
        
        for i in range(len(selected_jobs)):
            job = joblist[selected_jobs[i]]
            loaded_job = config["jobList"][i]

            # Since no argList is loaded as an empty list by ext_job
            arg_list_back_up = job["argList"]
            job["argList"] = empty_list_if_none(job["argList"])

            # Since name is set to default if none provided by ext_job
            name_back_up = job["name"]
            job["name"] = default_name_if_none(job["name"])

            for key in json_keywords:
                self.assertEqual(job[key], loaded_job[key])

            job["argList"] = arg_list_back_up
            job["name"] = name_back_up

    def test_no_jobs(self):
        with TestAreaContext("python/job_queue/forward_model_no_jobs"):
            forward_model = self.set_up_forward_model([])
            umask = 4
            global_args = SubstitutionList()
            forward_model.formatted_fprintf(
                    os.getcwd(),
                    global_args,
                    umask)

            self.verify_json_dump([], global_args, umask)

    def test_one_job(self):
        with TestAreaContext("python/job_queue/forward_model_one_job"):
            for i in range(len(joblist)):
                forward_model = self.set_up_forward_model([i])
                umask = 11
                global_args = SubstitutionList()
                forward_model.formatted_fprintf(
                    os.getcwd(),
                    global_args,
                    umask)

                self.verify_json_dump([i], global_args, umask)

    def run_all(self):
        forward_model = self.set_up_forward_model(range(len(joblist)))
        umask = 0
        global_args = SubstitutionList()
        forward_model.formatted_fprintf(
                os.getcwd(),
                global_args,
                umask)

        self.verify_json_dump(range(len(joblist)), global_args, umask)

    def test_all_jobs(self):
        with TestAreaContext("python/job_queue/forward_model_all"):
            self.run_all()

    def test_name_none(self):
        with TestAreaContext("python/job_queue/forward_model_no_name"):
            name_back_up = joblist[0]["name"]
            license_path_back_up = joblist[0]["license_path"]

            joblist[0]["name"] = None
            joblist[0]["license_path"] = os.path.join(
                    get_license_root_path(joblist[0]["license_path"]),
                    DEFAULT_NAME
                    )
            
            self.run_all()

            joblist[0]["name"] = name_back_up
            joblist[0]["license_path"] = license_path_back_up
 
    def test_various_null_fields(self):
        for key in [
            "target_file",
            "error_file",
            "start_file",
            "stdout",
            "stderr",
            "max_running_minutes",
            "argList",
            "environment",
            "stdin"
            ]:
                with TestAreaContext("python/job_queue/forward_model_none_" + key):
                    back_up = joblist[0][key]
                    joblist[0][key] = None
                    self.run_all()
                    joblist[0][key] = back_up
