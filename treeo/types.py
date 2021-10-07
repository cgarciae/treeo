import dataclasses
import typing as tp
from abc import ABCMeta

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
import typing_extensions as tpe

from treeo import utils

A = tp.TypeVar("A")
B = tp.TypeVar("B")


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


class FieldMetadata:
    node: bool
    kind: type
    opaque: tp.Union[bool, tp.Callable[[utils.Opaque, tp.Any], bool]]

    def __init__(
        self,
        node: bool,
        kind: type,
        opaque: tp.Union[bool, tp.Callable[[utils.Opaque, tp.Any], bool]],
    ):
        self.__dict__["node"] = node
        self.__dict__["kind"] = kind
        self.__dict__["opaque"] = opaque

    def update(self, **kwargs) -> "FieldMetadata":
        fields = vars(self).copy()
        fields.update(kwargs)
        return FieldMetadata(**fields)

    def __setattr__(self, name: str, value: tp.Any) -> None:
        raise AttributeError(f"FieldMetadata is immutable, cannot set {name}")


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
