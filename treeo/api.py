import functools
import inspect
import logging
import re
import typing as tp
from contextlib import contextmanager
from io import StringIO

import jax
import jax.numpy as jnp
import numpy as np
from attr import has

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
C = tp.TypeVar("C", bound=tp.Callable)
T = tp.TypeVar("T", bound="Tree")
Filter = tp.Union[
    tp.Type[tp.Any],
    tp.Callable[["FieldInfo"], bool],
]
F = tp.TypeVar("F", bound="tp.Callable")

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
        flatten_mode: Sets a new `FlattenMode` context for the operation.
    Returns:
        A new pytree with the filtered fields.

    """

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

    return obj


def merge(
    obj: A,
    other: A,
    *rest: A,
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
        flatten_mode: Sets a new `FlattenMode` context for the operation, if `None` the current context is used. If the current flatten context is `None` and `flatten_mode` is not passed then `FlattenMode.all_fields` is used.
        ignore_static: If `True`, bypasses static fields during the process and the statics fields for output are taken from the first input (`obj`).

    Returns:
        A new pytree with the updated values.
    """

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

    return obj


def map(
    f: tp.Callable,
    obj: A,
    *filters: Filter,
    flatten_mode: tp.Union[FlattenMode, str, None] = None,
    is_leaf: tp.Callable[[tp.Any], bool] = None,
    field_info: tp.Optional[bool] = False,
) -> A:
    """
    Applies a function to all leaves in a pytree using `jax.tree_map`, if `filters` are given then
    the function will be applied only to the subset of leaves that match the filters. For more information see
    [map's user guide](https://cgarciae.github.io/treeo/user-guide/api/map).


    Arguments:
        f: The function to apply to the leaves.
        obj: a pytree possibly containing `to.Tree`s.
        *filters: The filters used to select the leaves to which the function will be applied.
        flatten_mode: Sets a new `FlattenMode` context for the operation, if `None` the current context is used.
        add_field_info: Represent the leaves of the tree by a `FieldInfo` type. This enables values of the field such as
        kind and value to be used within the `map` function.

    Returns:
        A new pytree with the changes applied.
    """

    input_obj = obj

    has_filters = len(filters) > 0

    with _flatten_context(flatten_mode):
        if has_filters:
            new_obj = filter(obj, *filters)
        else:
            new_obj = obj

        # Conditionally build map function with, or without, the leaf nodes' field info.
        if field_info:
            with add_field_info():
                new_obj: A = jax.tree_map(f, new_obj, is_leaf=is_leaf)
        else:
            new_obj: A = jax.tree_map(f, new_obj, is_leaf=is_leaf)

        if has_filters:
            new_obj = merge(obj, new_obj)

    return new_obj


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
        obj = tree_m.apply(_remove_field_info_from_metadata, obj)

    return _to_dict(obj, private_fields, static_fields, type_info)


def _remove_field_info_from_metadata(obj: Tree):
    with tree_m._make_mutable_toplevel(obj):
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

    if hasattr(f, "_treeo_mutable"):
        raise ValueError(
            f"""Cannot make 'compact' a 'mutable' function, invert the order. If you are using it as a decorator, instead of e.g.
    
    @compact
    @mutable
    def {f.__name__}(self, ...):
    
use:

    @mutable
    @compact
    def {f.__name__}(self, ...):

"""
        )

    @functools.wraps(f)
    def wrapper(tree, *args, **kwargs):
        with tree_m._COMPACT_CONTEXT.compact(f, tree):
            return f(tree, *args, **kwargs)

    wrapper._treeo_compact = True

    return wrapper


def mutable(
    f: tp.Callable[..., A],
    *,
    toplevel_only: bool = False,
) -> tp.Callable[..., tp.Tuple[A, tp.Any]]:
    """
    A decorator that transforms a stateful function `f` that receives an Tree
    instance as a its first argument into a function that returns a tuple of the result and a Tree
    with the new state.

    This is useful for 2 reasons:
    * It transforms `f` into a pure function.
    * It allows `Immutable` Trees to perform inline field updates without getting `RuntimeError`s.

    Note that since the original object is not modified, `Immutable` instance remain in the end immutable.

    Example:

    ```python
    def accumulate_id(tree: MyTree, x: int) -> int:
        tree.n += x
        return x

    tree0 = MyTree(n=4)
    y, tree1 = mutable(accumulate_id)(tree0, 1)

    assert tree0.n == 4
    assert tree1.n == 5
    assert y == 1
    ```

    **Note**: Any `Tree`s that are found in the output of `f` are set to being
    immutable.

    Arguments:
        f: The function to be transformed.
        toplevel_only: If `True`, only the top-level object is made mutable.

    Returns:
        A function that returns a tuple of the result and a Tree with the new state.
    """

    f0 = f

    if inspect.ismethod(f):
        tree0 = f.__self__
        f = f.__func__
    elif isinstance(f, tree_m.Tree) and callable(f):
        tree0 = f
        f = f.__class__.__call__
    else:
        tree0 = None

    if tree0 is not None and not isinstance(tree0, Tree):
        name = f0.__name__ is hasattr(f0, "__name__") and f0.__class__.__name__
        raise TypeError(
            f"Invalid bounded method or callable '{name}', tried to infer unbouded function and instance, "
            f"expected a 'Tree' instance but '{type(tree0).__name__}' instead. Try using an unbounded class method instead."
        )

    @functools.wraps(f)
    def wrapper(tree, *args, **kwargs) -> tp.Tuple[A, tp.Any]:

        tree = tree_m.copy(tree)

        with tree_m.make_mutable(tree, toplevel_only=toplevel_only):
            output = f(tree, *args, **kwargs)

        def _make_output_immutable(a: Tree):
            tree_m._set_mutable(a, None)

        output = tree_m.apply(_make_output_immutable, output)

        return output, tree

    wrapper._treeo_mutable = True

    if tree0 is not None:

        @functools.wraps(f)
        def obj_wrapper(*args, **kwargs):
            return wrapper(tree0, *args, **kwargs)

        return obj_wrapper

    return wrapper


def toplevel_mutable(f: C) -> C:
    """
    A decorator that transforms a stateful function `f` that receives an Tree
    instance as a its first argument into a mutable function. It differs from `mutable`
    in the following ways:

    * It always applies mutability to the top-level object only.
    * `f` is expected to return the new state either as the only output or
        as the last element of a tuple.

    Example:

    ```python
    @dataclass
    class Child(to.Tree, to.Immutable):
        n: int = to.node()

    @dataclass
    def Parent(to.Tree, to.Immutable):
        child: Child

        @to.toplevel_mutable
        def update(self) -> "Parent":
            # self is currently mutable
            self.child = self.child.replace(n=self.child.n + 1) # but child is immutable (so we use replace)

            return self

    tree = Parent(child=Child(n=4))
    tree = tree.update()
    ```

    This behaviour is useful when the top-level tree mostly manipulates sub-trees that have well-defined
    immutable APIs, avoids explicitly run `replace` to propagate updates to the sub-trees and makes
    management of the top-level tree easier.

    **Note**: Any `Tree`s that are found in the output of `f` are set to being
    immutable, however the element is the to the same immutablity status as the
    input tree if they have the same type.

    Arguments:
        f: The function to be transformed.

    Returns:
        A function with top-level mutability.
    """

    f0 = f

    if inspect.ismethod(f):
        tree0 = f.__self__
        f = f.__func__
    elif isinstance(f, tree_m.Tree) and callable(f):
        tree0 = f
        f = f.__class__.__call__
    else:
        tree0 = None

    if tree0 is not None and not isinstance(tree0, Tree):
        name = f0.__name__ is hasattr(f0, "__name__") and f0.__class__.__name__
        raise TypeError(
            f"Invalid bounded method or callable '{name}', tried to infer unbouded function and instance, "
            f"expected a 'Tree' instance but '{type(tree0).__name__}' instead. Try using an unbounded class method instead."
        )

    @functools.wraps(f)
    def wrapper(tree: tree_m.Tree, *args, **kwargs):
        if not isinstance(tree, tree_m.Tree):
            raise TypeError(f"Expected 'Tree' type, got '{type(tree).__name__}'")

        output, _ = mutable(f, toplevel_only=True)(tree, *args, **kwargs)

        if isinstance(output, tuple):
            *ys, last = output
        else:
            ys = ()
            last = output

        if type(last) is type(tree):
            tree_m._set_mutable(last, tree._mutable)

        if isinstance(output, tuple):
            return (*ys, last)
        else:
            return last

    wrapper._treeo_mutable = True

    if tree0 is not None:

        @functools.wraps(f)
        def obj_wrapper(*args, **kwargs):
            return wrapper(tree0, *args, **kwargs)

        return obj_wrapper

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
