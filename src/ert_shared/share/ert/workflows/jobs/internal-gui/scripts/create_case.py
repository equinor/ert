from res.enkf import ErtScript


class CreateCaseJob(ErtScript):
    def run(self, case_name):
        ert = self.ert()
        ert.getEnkfFsManager().getFileSystem(case_name)
