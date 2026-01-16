import re

import pytest

from ert.plugins.plugin_manager import ErtPluginManager
from ert.shared._doc_utils.ert_jobs import _ErtDocumentation


@pytest.mark.parametrize(
    ("test_input", "expected_length"),
    [
        (
            {"JOB1": {}},
            1,
        ),
        (
            {
                "JOB1": {},
                "job2": {},
            },
            1,
        ),
        (
            {
                "JOB1": {},
                "JOB2": {},
            },
            2,
        ),
    ],
)
def test_divide_into_categories_all_default(test_input, expected_length):
    result = _ErtDocumentation._divide_into_categories(test_input)
    assert len(result["other"]["other"]) == expected_length


def test_divide_into_categories_lower_case_job():
    result = _ErtDocumentation._divide_into_categories({"job1": {}})
    assert result == {}


@pytest.mark.parametrize(
    ("test_input", "expected_category", "expected_sub_category"),
    [
        (
            {
                "JOB1": {
                    "category": "test.category.for.job",
                },
            },
            "test",
            "category",
        ),
        (
            {
                "JOB1": {
                    "category": "some_category.category.for.job",
                },
            },
            "some_category",
            "category",
        ),
        (
            {
                "JOB1": {},
            },
            "other",
            "other",
        ),
    ],
)
def test_divide_into_categories_main_category(
    test_input, expected_category, expected_sub_category
):
    result = _ErtDocumentation._divide_into_categories(test_input)
    assert expected_category in result
    assert expected_sub_category in result[expected_category]


@pytest.mark.parametrize(
    ("test_input", "expected_source_package"),
    [
        (
            {
                "JOB1": {
                    "source_package": "dummy",
                },
            },
            ["dummy"],
        ),
        (
            {
                "JOB1": {},
            },
            ["PACKAGE NOT PROVIDED"],
        ),
        (
            {
                "JOB1": {
                    "source_package": "dummy",
                },
                "JOB2": {},
            },
            ["dummy", "PACKAGE NOT PROVIDED"],
        ),
        (
            {
                "JOB1": {
                    "source_package": "dummy",
                },
                "JOB2": {"source_package": "example"},
            },
            ["dummy", "example"],
        ),
    ],
)
def test_divide_into_categories_job_source(test_input, expected_source_package):
    categories = _ErtDocumentation._divide_into_categories(test_input)
    result = [docs.job_source for docs in categories["other"]["other"]]
    assert expected_source_package == result


@pytest.mark.parametrize("fm_step", ErtPluginManager().forward_model_steps)
def test_that_forward_model_step_documentation_defaults_to_plugins_source_package(
    fm_step,
):
    # The test assumes that repr(fm_step) returns a string on format:
    # "<class 'fully.qualified.class.name'>"
    match_qualified_class_name = re.match(r"\<class '(.*?)'>", repr(fm_step))
    assert match_qualified_class_name, "Could not get qualified class name from fm_step"
    qualified_class_name = match_qualified_class_name[1]

    source_package = qualified_class_name.split(".")[0]
    documentation = fm_step.documentation()

    if documentation:
        assert documentation.source_package == source_package, (
            f"{qualified_class_name} documentation reports source_package="
            f"{documentation.source_package}, expected {source_package}"
        )
