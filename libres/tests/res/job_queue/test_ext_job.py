import os.path

from ecl.util.test import TestAreaContext
from tests import ResTest
from res.job_queue.ext_job import ExtJob
from res.config import ContentTypeEnum


def create_valid_config(config_file):
    with open(config_file, "w") as f:
        f.write("STDOUT null\n")
        f.write("STDERR null\n")
        f.write("EXECUTABLE script.sh\n")

    with open("script.sh", "w") as f:
        f.write("This is a script")


def create_upgraded_valid_config(config_file):
    with open(config_file, "w") as f:
        f.write("EXECUTABLE script.sh\n")
        f.write("MIN_ARG 2\n")
        f.write("MAX_ARG 7\n")
        f.write("ARG_TYPE 0 INT\n")
        f.write("ARG_TYPE 1 FLOAT\n")
        f.write("ARG_TYPE 2 STRING\n")
        f.write("ARG_TYPE 3 BOOL\n")
        f.write("ARG_TYPE 4 RUNTIME_FILE\n")
        f.write("ARG_TYPE 5 RUNTIME_INT\n")

    with open("script.sh", "w") as f:
        f.write("This is a script")


def create_config_missing_executable(config_file):
    with open(config_file, "w") as f:
        f.write("EXECUTABLE missing_script.sh\n")


def create_config_missing_EXECUTABLE(config_file):
    with open(config_file, "w") as f:
        f.write("EXECU missing_script.sh\n")


def create_config_executable_directory(config_file):
    with open(config_file, "w") as f:
        f.write("EXECUTABLE /tmp\n")


def create_config_foreign_file(config_file):
    with open(config_file, "w") as f:
        f.write("EXECUTABLE /etc/passwd\n")


class ExtJobTest(ResTest):
    def test_load_forward_model(self):
        with self.assertRaises(IOError):
            job = ExtJob("CONFIG_FILE", True)

        with TestAreaContext("python/job_queue/forward_model1"):
            create_valid_config("CONFIG")
            job = ExtJob("CONFIG", True)
            self.assertEqual(job.name(), "CONFIG")
            self.assertEqual(job.get_stdout_file(), None)
            self.assertEqual(job.get_stderr_file(), None)

            self.assertEqual(
                job.get_executable(), os.path.join(os.getcwd(), "script.sh")
            )
            self.assertTrue(os.access(job.get_executable(), os.X_OK))

            self.assertEqual(job.min_arg, -1)

            job = ExtJob("CONFIG", True, name="Job")
            self.assertEqual(job.name(), "Job")
            pfx = "ExtJob("
            self.assertEqual(pfx, repr(job)[: len(pfx)])

        with TestAreaContext("python/job_queue/forward_model1a"):
            create_upgraded_valid_config("CONFIG")
            job = ExtJob("CONFIG", True)
            self.assertEqual(job.min_arg, 2)
            self.assertEqual(job.max_arg, 7)
            argTypes = job.arg_types
            self.assertEqual(
                argTypes,
                [
                    ContentTypeEnum.CONFIG_INT,
                    ContentTypeEnum.CONFIG_FLOAT,
                    ContentTypeEnum.CONFIG_STRING,
                    ContentTypeEnum.CONFIG_BOOL,
                    ContentTypeEnum.CONFIG_RUNTIME_FILE,
                    ContentTypeEnum.CONFIG_RUNTIME_INT,
                    ContentTypeEnum.CONFIG_STRING,
                ],
            )

        with TestAreaContext("python/job_queue/forward_model2"):
            create_config_missing_executable("CONFIG")
            with self.assertRaises(ValueError):
                job = ExtJob("CONFIG", True)

        with TestAreaContext("python/job_queue/forward_model3"):
            create_config_missing_EXECUTABLE("CONFIG")
            with self.assertRaises(ValueError):
                job = ExtJob("CONFIG", True)

        with TestAreaContext("python/job_queue/forward_model4"):
            create_config_executable_directory("CONFIG")
            with self.assertRaises(ValueError):
                job = ExtJob("CONFIG", True)

        with TestAreaContext("python/job_queue/forward_model5"):
            create_config_foreign_file("CONFIG")
            with self.assertRaises(ValueError):
                job = ExtJob("CONFIG", True)

    def test_valid_args(self):
        arg_types = [
            ContentTypeEnum.CONFIG_FLOAT,
            ContentTypeEnum.CONFIG_INT,
            ContentTypeEnum.CONFIG_BOOL,
            ContentTypeEnum.CONFIG_STRING,
        ]
        run_arg_types = [
            ContentTypeEnum.CONFIG_RUNTIME_INT,
            ContentTypeEnum.CONFIG_RUNTIME_INT,
        ]

        arg_list = ["5.6", "65", "True", "car"]
        self.assertTrue(ExtJob.valid_args(arg_types, arg_list))
        arg_list2 = ["True", "True", "8", "car"]
        self.assertFalse(ExtJob.valid_args(arg_types, arg_list2))

        run_arg_list = ["Trjue", "76"]
        self.assertTrue(ExtJob.valid_args(run_arg_types, run_arg_list))
        self.assertFalse(ExtJob.valid_args(run_arg_types, run_arg_list, True))
