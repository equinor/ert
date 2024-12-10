from .parsing import SchemaItemType


def parse_arg_types_list(
    specified_arg_types: list[tuple[int, str]],
    specified_min_args: int,
    specified_max_args: int,
) -> list[SchemaItemType]:
    # First find the number of args
    specified_highest_arg_index = (
        max(index for index, _ in specified_arg_types)
        if len(specified_arg_types) > 0
        else -1
    )

    num_args = max(
        specified_highest_arg_index + 1,
        specified_max_args,
        specified_min_args,
    )

    arg_types_dict: dict[int, SchemaItemType] = {}

    for i, type_as_string in specified_arg_types:
        arg_types_dict[i] = _string_to_type(type_as_string)

    arg_types_list: list[SchemaItemType] = [
        arg_types_dict.get(i, SchemaItemType.STRING) for i in range(num_args)
    ]
    return arg_types_list


def _string_to_type(string: str) -> SchemaItemType:
    return SchemaItemType(string)
