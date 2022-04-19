##Performance tests
In this folder are tests that test performance with pytest-benchmark. 

##Fixture with example data
There is a fixture "poly_ran"
in the conftest.py file. This fixture is run once for each of the "cases_to_run", also defined in
conftest.py. If your test depends on the poly_ran fixture, each of the cases will become a 
separate parameter for your test, e.g. the test will potentially be run many times.

If you want to add another case, just add to the "cases_to_run" list in conftest.py

You can see an example of how to use this in combination with other parameters and 
pytest-benchmark in the test "test_dark_storage_performance.py"

###Using the fixture "poly_ran"
Just declare "poly_ran" in the argument list of the test. The parameter given to your test will be
a dict with information of what was run. It contains all the parameters for the make_poly_example()
funtion (see the tests/poly_template/README.md for this list), and in addition the folder where the
experiment ran and config file resides.

You should not use this fixture if you are going to change anything, as the fixture is shared
("session" scoped in pytest)
