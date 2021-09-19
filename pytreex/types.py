import dataclasses
import typing as tp
from abc import ABCMeta
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np

from pytreex import utils

A = tp.TypeVar("A")
B = tp.TypeVar("B")


class FieldMixin:
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
        lazy: bool = False,
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
            lazy=lazy,
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
        lazy: bool = False,
    ) -> tp.Any:
        return utils.node(
            default=default,
            kind=cls,
            default_factory=default_factory,
            init=init,
            repr=repr,
            hash=hash,
            compare=compare,
            lazy=lazy,
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
        lazy: bool = False,
    ) -> tp.Any:
        return cls.field(
            default=default,
            node=False,
            default_factory=default_factory,
            init=init,
            repr=repr,
            hash=hash,
            compare=compare,
            lazy=lazy,
        )


class TrivialPytree:
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


@dataclass
class FieldMetadata(TrivialPytree):
    node: bool
    kind: type


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
