##Performance tests
In this folder are tests that test performance with pytest-benchmark. When you add a test under this
folder it will be run automatically both in ci and daily on jenkins. The tests are set up to have a
small data set to run in ci, and a large to run daily. The daily job will fail when there are big
performance fluctuations from a baseline.

##Adding more tests and update baseline
Make sure you update the baseline after adding new tests, or changing their definition in any way,
so that the comparison can be done. This can be done by running the jenkins job, which will upload
the the new baseline as an artifact. Download and put the new file in the .benchmarks folder, next
to the old baseline.

Don't allow the performance to gradually change inadvertantly when adding a new baseline.

There is an argument COMPARE_VERSION to the jenkins job which controls the version to check agains,
this must be updated to use the new baseline.

##Fixture with example data
There is a fixture "template_config"
in the conftest.py file. This fixture is run once for each of the "cases_to_run", also defined in
conftest.py. If your test depends on the poly_ran fixture, each of the cases will become a 
separate parameter for your test, e.g. the test will potentially be run many times.

If you want to add another case, just add to the "cases_to_run" list in conftest.py

You can see an example of how to use this in combination with other parameters and 
pytest-benchmark in the test "test_dark_storage_performance.py"

###Using the fixture "template_config"
Just declare "template_config" in the argument list of the test. The parameter given to your test 
will be a dict with information of what was run. It contains all the parameters for the 
make_poly_example() funtion (see the tests/poly_template/README.md for this list), and in addition 
the folder where the experiment ran and config file resides.

You should not use this fixture if you are going to change anything, as the fixture is shared
("session" scoped in pytest)
