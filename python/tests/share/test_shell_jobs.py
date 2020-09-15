from res.enkf import ResConfig

from tests import ResTest


class TestSiteConfigShellJobs(ResTest):
    def test_shell_script_jobs_availability(self):
        config_file = self.createTestPath("local/simple_config/minimum_config")

        res_config = ResConfig(config_file)
        site_config = res_config.site_config

        installed_jobs = site_config.get_installed_jobs()
        fm_shell_jobs = {}
        for job in installed_jobs:
            exe = job.get_executable()
            if "shell_scripts" in exe:
                fm_shell_jobs[job.name().lower()] = exe

        list_from_content = res_config.ert_workflow_list
        wf_shell_jobs = {}
        for wf_name in list_from_content.getJobNames():
            exe = list_from_content.getJob(wf_name).executable()
            if exe and "shell_scripts" in exe:
                wf_shell_jobs[wf_name] = exe

        assert fm_shell_jobs == wf_shell_jobs

    def test_shell_script_jobs_names(self):
        config_file = self.createTestPath("local/simple_config/minimum_config")

        shell_job_names = [
            "delete_file",
            "delete_directory",
            "copy_directory",
            "make_symlink",
            "move_file",
            "make_directory",
            "careful_copy_file",
            "symlink",
            "copy_file",
        ]

        res_config = ResConfig(config_file)
        found_jobs = set()
        list_from_content = res_config.ert_workflow_list
        for wf_name in list_from_content.getJobNames():
            exe = list_from_content.getJob(wf_name).executable()
            if exe and "shell_scripts" in exe:
                assert wf_name in shell_job_names
                found_jobs.add(wf_name)

        assert len(shell_job_names) == len(found_jobs)
