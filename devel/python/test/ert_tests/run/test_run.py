from ert.test_run import TestRun
from ert_tests import ExtendedTestCase

    


class RunTest(ExtendedTestCase):

    def test_init(self):
        with self.assertRaises(IOError):
            TestRun("Does/notExist")
            
        tr = TestRun("test-data/local/run/config.txt")
        self.assertEqual( tr.config_file() , "test-data/local/run/config.txt")


    def test_cmd(self):
        tr = TestRun("test-data/local/run/config.txt")
        self.assertEqual( tr.ert_cmd , TestRun.ert_cmd )

        tr.ert_cmd = "/tmp/test"
        self.assertEqual( "/tmp/test" , tr.ert_cmd )


    def test_args(self):
        tr = TestRun("test-data/local/run/config.txt")
        self.assertEqual( tr.get_args() , [])

        tr.add_arg("-v")
        self.assertEqual( tr.get_args() , ["-v"])
        tr.add_arg("latest")
        self.assertEqual( tr.get_args() , ["-v" , "latest"])


    def test_workflows(self):
        tr = TestRun("test-data/local/run/config.txt")
        self.assertEqual( tr.get_workflows() , [])
        
        tr.add_workflow( "wf1" )
        tr.add_workflow( "wf2" )
        self.assertEqual( tr.get_workflows() , ["wf1" , "wf2"])

        
    def test_run_no_workflow(self):
        tr = TestRun("test-data/local/run/config.txt")
        with self.assertRaises(Exception):
            tr.run()

        tr.add_workflow("wf1")
        tr.start()
        
            
    def test_name(self):
        tr = TestRun("test-data/local/run/config.txt" , "Name")
        self.assertEqual( "Name" , tr.name )

        tr = TestRun("test-data/local/run/config.txt")
        self.assertEqual( "test-data.local.run.config.txt" , tr.name )
        

    def test_runpath(self):
        tr = TestRun("test-data/local/run/config.txt" , "Name")
        self.assertEqual( TestRun.path_prefix , tr.path_prefix )
        
