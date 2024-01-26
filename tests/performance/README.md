## Performance tests

The tests in this folder test performance with the use of `pytest-benchmark`.

The tests are run onprem in a komodo-releases workflow called run_ert_pytest_storage_benchmarks. It is either dispatched manually, or triggered by a schedule. The schedule is a cron job running every 12 hours, and it will run the performance tests if there has been a new commit added after the previous run. The job will fail when there are big performance fluctuations from a baseline, and will notify us in slack, and create a PR with the result added in the .benchmarks/ directory in the ert repository.


## Adding more tests and update baseline

Make sure you update the baseline after adding new tests, or changing their definition in any way, so that the comparison can be done. This can be done by running the Github Actions job, which will create a PR with the new result (if you check off the "save_on_success" option). The benchmark baseline version is set in the workflow as an env variable, and has to be changed there. This is to avoid having regression over time (in regards to the initial result).

Don't allow the performance to gradually change inadvertantly when adding a new baseline.

## Fixture with example data

There is a fixture `template_config` in the `ert/tests/performance_tests/conftest.py` file. This fixture is run once for each of the "cases_to_run", also defined in `conftest.py`. See below for more about this.

If your test depends on the `poly_ran` fixture, each of the cases will become a separate parameter for your test, e.g. the test will potentially be run many times.

If you want to add another case, just add to the "cases_to_run" list in `conftest.py`.

You can see an example of how to use this in combination with other parameters and  pytest-benchmark in `test_dark_storage_performance.py`.


### Using the fixture `template_config`

Declare `template_config` in the argument list of the test. The parameter given to your test will be a dict with information of what was run. It contains all the parameters for the  `make_poly_example()` function (see `tests/poly_template/README.md` for this list), and in addition the folder where the experiment ran and config file resides.

You should not use this fixture if you are going to change anything, as the fixture is shared ("session" scoped in pytest).
