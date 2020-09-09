from res.job_queue import ExtJob

from res.enkf import ResConfig, ErtWorkflowList, SiteConfig

from tests import ResTest


class TestSiteConfigShellJobs(ResTest):
    def test_shell_script_jobs_availability(self):
        config_file = self.createTestPath("local/simple_config/minimum_config")
        share_path = self.createSharePath("ert/shell_scripts")

        res_config = ResConfig(config_file)
        site_config: SiteConfig = res_config.site_config

        installed_jobs = site_config.get_installed_jobs()
        fm_shell_jobs = {}
        job: ExtJob
        for job in installed_jobs:
            exe = job.get_executable()
            if exe.startswith(share_path):
                fm_shell_jobs[job.name().lower()] = exe

        list_from_content: ErtWorkflowList = res_config.ert_workflow_list
        wf_shell_jobs = {}
        for wf_name in list_from_content.getJobNames():
            exe = list_from_content.getJob(wf_name).executable()
            if exe and exe.startswith(share_path):
                wf_shell_jobs[wf_name] = exe

        assert fm_shell_jobs == wf_shell_jobs
