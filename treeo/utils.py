import dataclasses
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
import typing_extensions as tpe

A = tp.TypeVar("A")
B = tp.TypeVar("B")

key = jax.random.PRNGKey
_pymap = map
_pyfilter = filter

OpaquePredicate = tp.Optional[tp.Callable[["Opaque", tp.Any], bool]]


@tpe.runtime_checkable
class ArrayLike(tpe.Protocol):
    shape: tp.Tuple[int, ...]
    dtype: np.dtype


class Opaque(tp.Generic[A]):
    def __init__(self, value: A, predicate: tp.Optional[OpaquePredicate]):
        self.value = value
        self.predicate = predicate

    def __repr__(self):
        return f"Hidden({self.value})"

    def __eq__(self, other):
        if self.predicate is not None:
            return self.predicate(self, other)
        else:
            # if both are Opaque and their values are of the same type
            if isinstance(other, Opaque) and type(self.value) == type(other.value):
                # if they are array-like also compare their shapes and dtypes
                if isinstance(self.value, ArrayLike):
                    other_value = tp.cast(ArrayLike, other.value)
                    return (
                        self.value.shape == other_value.shape
                        and self.value.dtype == other_value.dtype
                    )
                else:
                    # else they are equal
                    return True
            else:
                return False


def field(
    default=dataclasses.MISSING,
    *,
    node: bool,
    kind: type = type(None),
    default_factory=dataclasses.MISSING,
    init: bool = True,
    repr: bool = True,
    hash: tp.Optional[bool] = None,
    compare: bool = True,
    opaque: tp.Union[bool, OpaquePredicate] = False,
) -> tp.Any:

    return dataclasses.field(
        default=default,
        metadata={
            "node": node,
            "kind": kind,
            "opaque": opaque,
        },
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
    )


def node(
    default=dataclasses.MISSING,
    *,
    kind: type = type(None),
    default_factory=dataclasses.MISSING,
    init: bool = True,
    repr: bool = True,
    hash: tp.Optional[bool] = None,
    compare: bool = True,
    opaque: tp.Union[bool, OpaquePredicate] = False,
) -> tp.Any:
    return field(
        default=default,
        node=True,
        kind=kind,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        opaque=opaque,
    )


def static(
    default=dataclasses.MISSING,
    *,
    kind: type = type(None),
    default_factory=dataclasses.MISSING,
    init: bool = True,
    repr: bool = True,
    hash: tp.Optional[bool] = None,
    compare: bool = True,
    opaque: tp.Union[bool, OpaquePredicate] = False,
) -> tp.Any:
    return field(
        default,
        node=False,
        kind=kind,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        opaque=opaque,
    )


def _get_all_annotations(cls: type) -> tp.Dict[str, type]:
    d = {}
    for c in reversed(cls.mro()):
        if hasattr(c, "__annotations__"):
            d.update(**c.__annotations__)
    return d


def _get_all_vars(cls: type) -> tp.Dict[str, tp.Any]:
    d = {}
    for c in reversed(cls.mro()):
        if hasattr(c, "__dict__"):
            d.update(vars(c))
    return d


def _all_types(t: type) -> tp.Iterable[type]:
    return _pyfilter(lambda t: isinstance(t, tp.Type), _all_types_unfiltered(t))


def _all_types_unfiltered(t: type) -> tp.Iterable[type]:
    yield t

    if hasattr(t, "__args__"):
        yield t.__origin__

        for arg in t.__args__:
            yield from _all_types_unfiltered(arg)
