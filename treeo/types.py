import dataclasses
import typing as tp
from abc import ABCMeta

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np

from treeo import utils
import typing_extensions as tpe

A = tp.TypeVar("A")
B = tp.TypeVar("B")


class KindMixin:
    @classmethod
    def field(
        cls,
        default=dataclasses.MISSING,
        *,
        node: bool,
        default_factory=dataclasses.MISSING,
        init: bool = True,
        repr: bool = True,
        hash: tp.Optional[bool] = None,
        compare: bool = True,
        opaque: bool = False,
        opaque_is_equal: tp.Optional[tp.Callable[[utils.Opaque, tp.Any], bool]] = None,
    ) -> tp.Any:
        return utils.field(
            default=default,
            node=node,
            kind=cls,
            default_factory=default_factory,
            init=init,
            repr=repr,
            hash=hash,
            compare=compare,
            opaque=opaque,
            opaque_is_equal=opaque_is_equal,
        )

    @classmethod
    def node(
        cls,
        default=dataclasses.MISSING,
        *,
        default_factory=dataclasses.MISSING,
        init: bool = True,
        repr: bool = True,
        hash: tp.Optional[bool] = None,
        compare: bool = True,
        opaque: bool = False,
        opaque_is_equal: tp.Optional[tp.Callable[[utils.Opaque, tp.Any], bool]] = None,
    ) -> tp.Any:
        return utils.node(
            default=default,
            kind=cls,
            default_factory=default_factory,
            init=init,
            repr=repr,
            hash=hash,
            compare=compare,
            opaque=opaque,
            opaque_is_equal=opaque_is_equal,
        )

    @classmethod
    def static(
        cls,
        default=dataclasses.MISSING,
        default_factory=dataclasses.MISSING,
        init: bool = True,
        repr: bool = True,
        hash: tp.Optional[bool] = None,
        compare: bool = True,
        opaque: bool = False,
        opaque_is_equal: tp.Optional[tp.Callable[[utils.Opaque, tp.Any], bool]] = None,
    ) -> tp.Any:
        return cls.field(
            default=default,
            node=False,
            default_factory=default_factory,
            init=init,
            repr=repr,
            hash=hash,
            compare=compare,
            opaque=opaque,
            opaque_is_equal=opaque_is_equal,
        )


class _TrivialPytree:
    def tree_flatten(self):
        tree = vars(self)
        children = (tree,)
        return (children, ())

    @classmethod
    def tree_unflatten(cls, aux, children):
        (tree,) = children

        obj = cls.__new__(cls)
        obj.__dict__.update(tree)

        return obj

    def __init_subclass__(cls):
        jax.tree_util.register_pytree_node_class(cls)


@dataclasses.dataclass
class FieldMetadata(_TrivialPytree):
    node: bool
    kind: type
    opaque: bool
    opaque_is_equal: tp.Optional[tp.Callable[[utils.Opaque, tp.Any], bool]]


@jax.tree_util.register_pytree_node_class
class Nothing:
    def tree_flatten(self):
        return (), None

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls()

    def __repr__(self) -> str:
        return "Nothing"

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Nothing)


NOTHING = Nothing()


class Missing:
    pass


MISSING = Missing()


class Hashable(tp.Generic[A]):
    """A hashable immutable wrapper around non-hashable values"""

    value: A

    def __init__(self, value: A):
        self.__dict__["value"] = value

    def __setattr__(self, name: str, value: tp.Any) -> None:
        raise AttributeError(f"Hashable is immutable")
