import functools
import logging
import re
import typing as tp
from contextlib import contextmanager
from io import StringIO

import jax
import jax.numpy as jnp
import numpy as np

import treeo.tree as tree_m
from treeo import types, utils
from treeo.tree import FieldInfo, FlattenMode, Tree

try:
    from rich.console import Console
    from rich.text import Text
except ImportError:
    Text = None
    Console = None

RICH_WARNING_COUNT = 0

A = tp.TypeVar("A")
B = tp.TypeVar("B")
T = tp.TypeVar("T", bound="Tree")
Filter = tp.Union[
    tp.Type[tp.Any],
    tp.Callable[["FieldInfo"], bool],
]

PAD_START = r"__PAD_START__"
PAD_END = r"__PAD_END__"
FIND_PAD = re.compile(f"{PAD_START}(.*){PAD_END}")
LEAF_TYPES = (types.Nothing, type(None))


# --------------------------------------------------
# functions
# --------------------------------------------------


def filter(
    obj: A,
    *filters: Filter,
    inplace: bool = False,
    flatten_mode: tp.Union[FlattenMode, str, None] = None,
) -> A:
    """
    The `filter` function allows you to select a subtree by filtering based on a predicate or `kind` type,
    leaves that pass all filters are kept, the rest are set to `Nothing`. For more information see
    [filter's user guide](https://cgarciae.github.io/treeo/user-guide/api/filter).



    Arguments:
        obj: A pytree (possibly containing `to.Tree`s) to be filtered.
        *filters: Types to filter by, membership is determined by `issubclass`, or
            callables that take in a `FieldInfo` and return a `bool`.
        inplace: If `True`, the input `obj` is mutated and returned.
        flatten_mode: Sets a new `FlattenMode` context for the operation.
    Returns:
        A new pytree with the filtered fields. If `inplace` is `True`, `obj` is returned.

    """
    if inplace and not hasattr(obj, "__dict__"):
        raise ValueError(
            f"Cannot filter inplace on objects with no __dict__ property, got {obj}"
        )

    input_obj = obj

    filters = tuple(
        _get_kind_filter(f) if isinstance(f, tp.Type) else f for f in filters
    )

    def apply_filters(info: tp.Any) -> tp.Any:
        if not isinstance(info, FieldInfo):
            info = FieldInfo(
                name=None,
                value=info,
                kind=type(None),
                module=None,
            )
        assert isinstance(info, FieldInfo)

        return info.value if all(f(info) for f in filters) else types.NOTHING

    with tree_m._CONTEXT.update(add_field_info=True), _flatten_context(flatten_mode):
        obj = jax.tree_map(apply_filters, obj)

    if inplace:
        input_obj.__dict__.update(obj.__dict__)
        return input_obj
    else:
        return obj


def merge(
    obj: A,
    other: A,
    *rest: A,
    inplace: bool = False,
    flatten_mode: tp.Union[FlattenMode, str, None] = None,
    ignore_static: bool = False,
) -> A:
    """
    Creates a new Tree with the same structure but its values merged based on the values from the incoming Trees. For more information see
    [merge's user guide](https://cgarciae.github.io/treeo/user-guide/api/merge).

    Arguments:
        obj: Main pytree to merge.
        other: The pytree first to get the values to merge with.
        *rest: Additional pytree to perform the merge in order from left to right.
        inplace: If `True`, the input `obj` is mutated and returned.
        flatten_mode: Sets a new `FlattenMode` context for the operation, if `None` the current context is used. If the current flatten context is `None` and `flatten_mode` is not passed then `FlattenMode.all_fields` is used.
        ignore_static: If `True`, bypasses static fields during the process and the statics fields for output are taken from the first input (`obj`).

    Returns:
        A new pytree with the updated values. If `inplace` is `True`, `obj` is returned.
    """

    if inplace and not hasattr(obj, "__dict__"):
        raise TypeError(
            f"Cannot update inplace on objects with no __dict__ property, got {obj}"
        )

    if flatten_mode is None and tree_m._CONTEXT.flatten_mode is None:
        flatten_mode = FlattenMode.all_fields

    input_obj = obj

    def merge_fn(*xs):
        for x in reversed(xs):
            if not isinstance(x, types.Nothing):
                return x
        return types.NOTHING

    tree_map_fn = _looser_tree_map if ignore_static else jax.tree_map

    with _flatten_context(flatten_mode):
        obj = tree_map_fn(
            merge_fn,
            obj,
            other,
            *rest,
            is_leaf=lambda x: isinstance(x, LEAF_TYPES),
        )

    if inplace:
        input_obj.__dict__.update(obj.__dict__)
        return input_obj
    else:
        return obj


def map(
    f: tp.Callable,
    obj: A,
    *filters: Filter,
    inplace: bool = False,
    flatten_mode: tp.Union[FlattenMode, str, None] = None,
    is_leaf: tp.Callable[[tp.Any], bool] = None,
) -> A:
    """
    Applies a function to all leaves in a pytree using `jax.tree_map`, if `filters` are given then
    the function will be applied only to the subset of leaves that match the filters. For more information see
    [map's user guide](https://cgarciae.github.io/treeo/user-guide/api/map).


    Arguments:
        f: The function to apply to the leaves.
        obj: a pytree possibly containing `to.Tree`s.
        *filters: The filters used to select the leaves to which the function will be applied.
        inplace: If `True`, the input `obj` is mutated and returned.
        flatten_mode: Sets a new `FlattenMode` context for the operation, if `None` the current context is used.

    Returns:
        A new pytree with the changes applied. If `inplace` is `True`, the input `obj` is returned.
    """
    if inplace and not hasattr(obj, "__dict__"):
        raise ValueError(
            f"Cannot map inplace on objects with no __dict__ property, got {obj}"
        )

    input_obj = obj

    has_filters = len(filters) > 0

    with _flatten_context(flatten_mode):
        if has_filters:
            new_obj = filter(obj, *filters)
        else:
            new_obj = obj

        new_obj: A = jax.tree_map(f, new_obj, is_leaf=is_leaf)

        if has_filters:
            new_obj = merge(obj, new_obj)

    if inplace:
        input_obj.__dict__.update(new_obj.__dict__)
        return input_obj
    else:
        return new_obj


def apply(f: tp.Callable[..., None], obj: A, *rest: A, inplace: bool = False) -> A:
    """
    Applies a function to all `to.Tree`s in a Pytree. Works very similar to `jax.tree_map`,
    but its values are `to.Tree`s instead of leaves, also `f` should apply the changes inplace to Tree object.

    If `inplace` is `False`, a copy of the first object is returned with the changes applied.
    The `rest` of the objects are always copied.

    Arguments:
        f: The function to apply.
        obj: a pytree possibly containing Trees.
        *rest: additional pytrees.
        inplace: If `True`, the input `obj` is mutated.

    Returns:
        A new pytree with the updated Trees or the same input `obj` if `inplace` is `True`.
    """
    rest = tree_m.copy(rest)

    if not inplace:
        obj = tree_m.copy(obj)

    objs = (obj,) + rest

    def nested_fn(obj, *rest):
        if isinstance(obj, Tree):
            apply(f, obj, *rest, inplace=True)

    jax.tree_map(
        nested_fn,
        *objs,
        is_leaf=lambda x: isinstance(x, Tree) and not x in objs,
    )

    if isinstance(obj, Tree):
        f(obj, *rest)

    return obj


def to_dict(
    obj: tp.Any,
    *,
    private_fields: bool = False,
    static_fields: bool = True,
    type_info: bool = False,
    field_info: bool = False,
) -> tp.Any:

    if field_info:
        with add_field_info(), flatten_mode(FlattenMode.all_fields):
            flat, treedef = jax.tree_flatten(obj)

        obj = jax.tree_unflatten(treedef, flat)
        obj = apply(_remove_field_info_from_metadata, obj)

    return _to_dict(obj, private_fields, static_fields, type_info)


def _remove_field_info_from_metadata(obj: Tree):

    obj._field_metadata = jax.tree_map(
        lambda x: x.value if isinstance(x, FieldInfo) else x,
        obj._field_metadata,
    )


def _to_dict(
    obj: tp.Any, private_fields: bool, static_fields: bool, type_info: bool
) -> tp.Any:

    if isinstance(obj, Tree):
        fields = vars(obj).copy()

        if not private_fields:
            fields = {k: v for k, v in fields.items() if not k.startswith("_")}

        if not static_fields:
            fields = {k: v for k, v in fields.items() if obj._field_metadata[k].node}

        fields = {
            k: _to_dict(v, private_fields, static_fields, type_info)
            for k, v in fields.items()
        }

        if type_info:
            fields["__type__"] = type(obj)

        return fields
    elif isinstance(obj, tp.Mapping):
        output = {
            k: _to_dict(v, private_fields, static_fields, type_info)
            for k, v in obj.items()
        }

        if type_info:
            output["__type__"] = type(obj)

        return output
    elif isinstance(obj, tp.Sequence) and not isinstance(obj, str):
        output = [_to_dict(v, private_fields, static_fields, type_info) for v in obj]

        if type_info:
            output.append(type(obj))

        return output
    else:
        return obj


def to_string(
    obj: tp.Any,
    private_fields: bool = False,
    static_fields: bool = True,
    color: bool = False,
) -> str:
    """
    Converts a pytree to a string representation.

    Arguments:
        obj: The pytree to convert.
        private_fields: If `True`, private fields are included.
        static_fields: If `True`, static fields are included.

    Returns:
        A string representation of the pytree.
    """
    dict_ = to_dict(
        obj,
        private_fields=private_fields,
        static_fields=static_fields,
        type_info=True,
        field_info=True,
    )
    global RICH_WARNING_COUNT
    rep = _to_string(dict_, level=0, inline=False, color=color, space="    ")
    rep = _add_padding(rep)

    if color:
        if Console is None or Text is None:
            if RICH_WARNING_COUNT < 1:
                RICH_WARNING_COUNT += 1
                logging.warning(
                    f"'rich' library not available, install `rich` to get colors."
                )
        else:
            rep = _get_rich_repr(Text.from_markup(rep))

    return rep


def _to_string(
    obj: tp.Any,
    *,
    level: int,
    inline: bool,
    color: bool,
    space: str,
) -> str:

    indent_level = space * level

    DIM = "[dim]" if color else ""
    END = "[/dim]" if color else ""

    if isinstance(obj, tp.Mapping):
        obj_type = obj["__type__"]
        body = [
            indent_level
            + space
            + f"{field}: {_to_string(value, level=level + 1, inline=True, color=color, space=space)},"
            for field, value in obj.items()
            if field != "__type__"
        ]
        body_str = "\n".join(body)
        type_str = f"{obj_type.__name__}" if inline else obj_type.__name__

        if len(obj) > 1:  # zero fields excluding __type__
            return f"{type_str} {{\n{body_str}\n{indent_level}}}"
        else:
            return f"{type_str} {{}}"

        # return f"\n{body_str}"
    elif isinstance(obj, tp.Sequence) and not isinstance(obj, str):
        obj_type = obj[-1]
        body = [
            indent_level
            + space
            + f"{_to_string(value, level=level + 1, inline=False, color=color, space=space)},"
            for i, value in enumerate(obj[:-1])
        ]
        body_str = "\n".join(body)
        type_str = f"{obj_type.__name__}" if inline else obj_type.__name__

        if len(obj) > 1:  # zero fields excluding __type__
            return f"{type_str} [\n{body_str}\n{indent_level}]"
        else:
            return f"{type_str} []"

    elif isinstance(obj, FieldInfo):
        value = obj.value
        kind_name = obj.kind.__name__ if obj.kind != type(None) else ""

        if isinstance(value, (np.ndarray, jnp.ndarray)):

            value_type = type(value)
            type_module = value_type.__module__.split(".")[0]
            value_rep = (
                f"{type_module}.{value_type.__name__}({value.shape}, {value.dtype})"
            )
        elif isinstance(value, str):
            value_rep = f'"{value}"'
        else:
            value_rep = str(value)

        return (
            f"{value_rep}{PAD_START}{DIM}{kind_name}{END}{PAD_END}"
            if kind_name
            else value_rep
        )

    else:
        return str(obj)


def in_compact() -> bool:
    """
    Returns:
        `True` if current inside a function decorated with `@compact`.
    """
    return tree_m._COMPACT_CONTEXT.in_compact


# ---------------------------------------------------------------
# Context Managers
# ---------------------------------------------------------------


@contextmanager
def add_field_info():
    """
    A context manager that makes `Tree`s produce leaves as `FieldInfo` when flattening.
    """
    with tree_m._CONTEXT.update(add_field_info=True):
        yield


@contextmanager
def flatten_mode(mode: tp.Optional[tp.Union[FlattenMode, str]]):
    """
    A context manager that defines how `Tree`s are flattened. Options are:

    * `'normal'`: Fields are selected as nodes as declared in the class definition (default behavior).
    * `'all_fields'`: All fields are treated as nodes during flattening.
    * `'no_fields'`: All fields are treated as static, `Tree`s produce no leaves.
    * `None`: Context is not changed, current flatten mode is preserved.

    Example:

    ```python
    @dataclass
    class MyTree(Tree):
        x: int # static
        y: int = to.node()

    tree = MyTree(x=1, y=3)

    jax.tree_map(lambda x: x * 2, tree) # MyTree(x=1, y=6)

    with flatten_mode('all_fields'):
        jax.tree_map(lambda x: x + 1, tree) # MyTree(x=2, y=6)
    ```

    Arguments:
        mode: The new flatten mode.
    """
    if mode is not None:
        if isinstance(mode, str):
            mode = FlattenMode(mode)

        with tree_m._CONTEXT.update(flatten_mode=mode):
            yield
    else:
        yield


# alias for internal use
_flatten_context = flatten_mode


# ---------------------------------------------------------------
# decorators
# ---------------------------------------------------------------


def compact(f):
    """
    A decorator that enable the definition of Tree subnodes at runtime.
    """

    @functools.wraps(f)
    def wrapper(tree, *args, **kwargs):
        with tree_m._COMPACT_CONTEXT.compact(f, tree):
            return f(tree, *args, **kwargs)

    wrapper._treeo_compact = True

    return wrapper


# ---------------------------------------------------------------
# utils
# ---------------------------------------------------------------


def _looser_tree_map(
    f: tp.Callable[..., tp.Any],
    tree: tp.Any,
    *rest: tp.Any,
    is_leaf: tp.Optional[tp.Callable[[tp.Any], bool]] = None,
) -> tp.Any:
    jax.tree_map
    leaves, treedef = jax.tree_flatten(
        tree,
        is_leaf=is_leaf,
    )
    all_leaves = [leaves] + [jax.tree_flatten(r, is_leaf=is_leaf)[0] for r in rest]

    n_leaves = len(leaves)
    assert all(len(l) == n_leaves for l in all_leaves)

    return treedef.unflatten(f(*xs) for xs in zip(*all_leaves))


def _get_kind_filter(
    t: type,
) -> tp.Callable[[FieldInfo], bool]:
    def _filter(info: FieldInfo) -> bool:
        return (
            info.kind is not None
            and isinstance(t, tp.Type)
            and issubclass(info.kind, t)
        )

    return _filter


def _get_rich_repr(table):
    assert Console is not None
    f = StringIO()
    console = Console(file=f, force_terminal=True)
    console.print(table)

    return f.getvalue()


def _add_padding(text):

    space = " "
    lines = text.split("\n")
    padded_info = []
    for i in range(len(lines)):
        match = FIND_PAD.search(lines[i])
        if match:
            lines[i] = FIND_PAD.sub("", lines[i])
            padded_info.append(match[1])
        else:
            padded_info.append("")
    lenghts = [len(line) for line in lines]
    max_length = max(lenghts)

    text = "\n".join(
        line + space * (max_length - length) + f"    {info}" if info else line
        for line, length, info in zip(lines, lenghts, padded_info)
    )

    return text
