"""
This module adds the SQLAlchemyMutation class, a graphene.Mutation whose
output mirrors an SQLAlchemyObjectType. This allows us to create mutation that behave
"""


from collections import OrderedDict
from typing import Any, Type, Iterable, Callable, Optional, TYPE_CHECKING, Dict

from graphene_sqlalchemy import SQLAlchemyObjectType as _SQLAlchemyObjectType
from graphene.types.mutation import MutationOptions
from graphene.types.utils import yank_fields_from_attrs
from graphene.types.interface import Interface
from graphene.utils.get_unbound_function import get_unbound_function
from graphene.utils.props import props
from graphene.types.objecttype import ObjectTypeOptions
from graphene import ObjectType, Field, Interface
from graphql import ResolveInfo


__all__ = ["SQLAlchemyObjectType", "SQLAlchemyMutation"]


if TYPE_CHECKING:
    from graphene_sqlalchemy.types.argument import Argument
    import graphene.types.field


class SQLAlchemyObjectType(_SQLAlchemyObjectType):
    class Meta:
        abstract = True

    def resolve_id(self, info: ResolveInfo) -> str:
        return str(self.id)


class SQLAlchemyMutation(SQLAlchemyObjectType):
    """
    Object Type Definition (mutation field), appropriated from graphene.Mutation

    SQLAlchemyMutation is a convenience type that helps us build a Field which
    takes Arguments and returns a mutation Output SQLAlchemyObjectType.

    Meta class options (optional):
        resolver (Callable resolver method): Or ``mutate`` method on Mutation class. Perform data
            change and return output.
        arguments (Dict[str, graphene.Argument]): Or ``Arguments`` inner class with attributes on
            Mutation class. Arguments to use for the mutation Field.
        name (str): Name of the GraphQL type (must be unique in schema). Defaults to class
            name.
        description (str): Description of the GraphQL type in the schema. Defaults to class
            docstring.
        interfaces (Iterable[graphene.Interface]): GraphQL interfaces to extend with the payload
            object. All fields from interface will be included in this object's schema.
        fields (Dict[str, graphene.Field]): Dictionary of field name to Field. Not recommended to
            use (prefer class attributes or ``Meta.output``).

    """

    class Meta:
        abstract = True

    @classmethod
    def __init_subclass_with_meta__(
        cls,
        interfaces: Iterable[Type[Interface]] = (),
        resolver: Callable = None,
        arguments: Dict[str, "Argument"] = None,
        _meta: Optional[ObjectTypeOptions] = None,
        **options: Any,
    ) -> None:
        if not _meta:
            _meta = MutationOptions(cls)

        fields = {}

        for interface in interfaces:
            assert issubclass(interface, Interface), (
                'All interfaces of {} must be a subclass of Interface. Received "{}".'
            ).format(cls.__name__, interface)
            fields.update(interface._meta.fields)

        fields = OrderedDict()
        for base in reversed(cls.__mro__):
            fields.update(yank_fields_from_attrs(base.__dict__, _as=Field))

        if not arguments:
            input_class = getattr(cls, "Arguments", None)

            if input_class:
                arguments = props(input_class)
            else:
                arguments = {}

        if not resolver:
            mutate = getattr(cls, "mutate", None)
            assert mutate, "All mutations must define a mutate method in it"
            resolver = get_unbound_function(mutate)

        if _meta.fields:
            _meta.fields.update(fields)
        else:
            _meta.fields = fields

        _meta.interfaces = interfaces
        _meta.resolver = resolver
        _meta.arguments = arguments

        super(SQLAlchemyMutation, cls).__init_subclass_with_meta__(
            _meta=_meta, **options
        )

    @classmethod
    def Field(
        cls,
        name: Optional[str] = None,
        description: Optional[str] = None,
        deprecation_reason: Optional[str] = None,
        required: bool = False,
        **kwargs: Any,
    ) -> "graphene.types.field.Field":
        """Mount instance of mutation Field."""
        return Field(
            cls,
            args=cls._meta.arguments,
            resolver=cls._meta.resolver,
            name=name,
            description=description or cls._meta.description,
            deprecation_reason=deprecation_reason,
            required=required,
            **kwargs,
        )
