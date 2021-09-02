from typing import List
import pytest


def test_openapi(ert_storage_app, dark_storage_app):
    """
    Test that the openapi.json of Dark Storage is identical to ERT Storage
    """
    expect = ert_storage_app.openapi()
    actual = dark_storage_app.openapi()

    # Remove textual data (descriptions and such) from ERT Storage's API.
    def _remove_text(data):
        if isinstance(data, dict):
            return {
                key: _remove_text(val)
                for key, val in data.items()
                if key not in ("description", "examples")
            }
        return data

    assert _remove_text(expect) == _remove_text(actual)


def test_graphql(env):
    from ert_storage.graphql import schema as ert_schema
    from ert_shared.dark_storage.graphql import schema as dark_schema
    from graphql import print_schema

    def _sort_schema(schema: str) -> str:
        """
        Assuming that each block is separated by an empty line, we sort the contents
        so that the order is irrelevant
        """
        sorted_blocks: List[str] = []
        for block in schema.split("\n\n"):
            lines = block.splitlines()
            if len(lines) == 1:  # likely a lone "Scalar SomeType"
                sorted_blocks.append(block)
                continue
            body = sorted(
                line for line in lines[1:-1] if "Pk:" not in line and " pk:" not in line
            )
            sorted_blocks.append("\n".join([lines[0], *body, lines[-1]]))
        return "\n\n".join(sorted_blocks)

    expect = _sort_schema(print_schema(ert_schema))
    actual = _sort_schema(print_schema(dark_schema))

    assert expect == actual
