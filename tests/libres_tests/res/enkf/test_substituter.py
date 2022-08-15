from pathlib import Path

from ert._c_wrappers.enkf import EnKFMain
from ert._c_wrappers.enkf.substituter import Substituter


def test_global_substitution():
    substituter = Substituter({"<case_name>": "case1"})

    assert (
        substituter.substitute("The case is: <case_name>", 0, 0) == "The case is: case1"
    )


def test_add_local_substitution():
    substituter = Substituter()

    substituter.add_substitution("<GEO_ID>", "id:00", 0, 0)
    substituter.add_substitution("<GEO_ID>", "id:01", 1, 0)

    assert (
        substituter.substitute("The geo id for <ITER>, <IENS> is: <GEO_ID>", 0, 0)
        == "The geo id for 0, 0 is: id:00"
    )
    assert (
        substituter.substitute("The geo id for <ITER>, <IENS> is: <GEO_ID>", 1, 0)
        == "The geo id for 0, 1 is: id:01"
    )
    assert (
        substituter.substitute("The geo id for <ITER>, <IENS> is: <GEO_ID>", 2, 2)
        == "The geo id for 2, 2 is: <GEO_ID>"
    )


def test_add_global_substitution():
    substituter = Substituter()

    substituter.add_global_substitution("<RUNPATH_FILE>", "/path/file")

    assert (
        substituter.substitute("The file is: <RUNPATH_FILE>", 0, 0)
        == "The file is: /path/file"
    )


def test_get_substitution_list():
    substituter = Substituter({"<GLOBAL>": "global"})

    assert substituter.get_substitutions(0, 1) == {
        "<GLOBAL>": "global",
        "<IENS>": "0",
        "<ITER>": "1",
    }


def test_substitution_workflow_integration(setup_case):
    res_config = setup_case("local/config/workflows", "config")
    ert = EnKFMain(res_config)
    current_filesystem = ert.getEnkfFsManager().getCurrentFileSystem()

    ert.getWorkflowList()["MAGIC_PRINT"].run(
        ert, context=ert.getWorkflowList().getContext()
    )

    assert (
        Path("magic-list.txt").read_text() == f"{current_filesystem.getCaseName()}"
        f"\nMagicAllTheWayToWorkFlow\n"
    )
