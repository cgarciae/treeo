import dataclasses
import inspect
import re
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
import typing_extensions as tpe

A = tp.TypeVar("A")
B = tp.TypeVar("B")

_pymap = map
_pyfilter = filter


@tpe.runtime_checkable
class ArrayLike(tpe.Protocol):
    shape: tp.Tuple[int, ...]
    dtype: np.dtype


def field(
    default: tp.Any = dataclasses.MISSING,
    *,
    node: bool,
    kind: type = type(None),
    default_factory: tp.Any = dataclasses.MISSING,
    init: bool = True,
    repr: bool = True,
    hash: tp.Optional[bool] = None,
    compare: bool = True,
) -> tp.Any:

    return dataclasses.field(
        default=default,
        metadata={
            "node": node,
            "kind": kind,
        },
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
    )


def node(
    default: tp.Any = dataclasses.MISSING,
    *,
    kind: type = type(None),
    default_factory: tp.Any = dataclasses.MISSING,
    init: bool = True,
    repr: bool = True,
    hash: tp.Optional[bool] = None,
    compare: bool = True,
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
    )


def static(
    default: tp.Any = dataclasses.MISSING,
    *,
    kind: type = type(None),
    default_factory: tp.Any = dataclasses.MISSING,
    init: bool = True,
    repr: bool = True,
    hash: tp.Optional[bool] = None,
    compare: bool = True,
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


def _unique_name(
    names: tp.Set[str],
    name: str,
):

    if name in names:

        match = re.match(r"(.*?)(\d*)$", name)
        assert match is not None

        name = match[1]
        num_part = match[2]

        i = int(num_part) if num_part else 2
        str_template = f"{{name}}{{i:0{len(num_part)}}}"

        while str_template.format(name=name, i=i) in names:
            i += 1

        name = str_template.format(name=name, i=i)

    names.add(name)
    return name


def _unique_names(
    names: tp.Iterable[str],
    *,
    existing_names: tp.Optional[tp.Set[str]] = None,
) -> tp.Iterable[str]:
    if existing_names is None:
        existing_names = set()

    for name in names:
        yield _unique_name(existing_names, name)


def _lower_snake_case(s: str) -> str:
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()
    parts = s.split("_")
    output_parts = []

    for i in range(len(parts)):
        if i == 0 or len(parts[i - 1]) > 1:
            output_parts.append(parts[i])
        else:
            output_parts[-1] += parts[i]

    return "_".join(output_parts)


def _get_name(obj) -> str:
    if hasattr(obj, "name") and obj.name:
        return obj.name
    elif hasattr(obj, "__name__") and obj.__name__:
        return _lower_snake_case(obj.__name__)
    elif hasattr(obj, "__class__") and obj.__class__.__name__:
        return _lower_snake_case(obj.__class__.__name__)
    else:
        raise ValueError(f"Could not get name for: {obj}")


def _safe_update_fields_from(obj, updates):
    for field, value in updates.__dict__.items():
        setattr(obj, field, value)
    return obj


def _get_unbound_method(obj: object, method: tp.Union[str, tp.Callable]) -> tp.Callable:

    if isinstance(method, str) and not hasattr(obj, method):
        raise TypeError(f"class '{type(obj).__name__}' has no attribute {method}")

    return (
        getattr(obj.__class__, method)
        if isinstance(method, str)
        else method.__func__
        if inspect.ismethod(method)
        else method
    )
