import dataclasses
import enum
import functools
import inspect
import io
import threading
import typing as tp
from abc import ABCMeta
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np

from treeo import types, utils

A = tp.TypeVar("A")
B = tp.TypeVar("B")
T = tp.TypeVar("T", bound="Tree")
Filter = tp.Union[
    tp.Type[tp.Any],
    tp.Callable[["FieldInfo"], bool],
]

PAD = r"{pad}"
LEAF_TYPES = (types.Nothing, type(None))


class FlattenMode(enum.Enum):
    no_fields = enum.auto()
    normal = enum.auto()
    all_fields = enum.auto()


@dataclass
class _Context(threading.local):
    add_field_info: bool = False
    flatten_mode: tp.Optional[FlattenMode] = None

    def __enter__(self):
        global _CONTEXT
        self._old_context = _CONTEXT
        _CONTEXT = self

    def __exit__(self, *args):
        global _CONTEXT
        _CONTEXT = self._old_context

    @contextmanager
    def update(self, **kwargs):
        fields = vars(self).copy()
        fields.pop("_old_context", None)
        fields.update(kwargs)

        with _Context(**fields):
            yield


_CONTEXT = _Context()


class FieldInfo:
    def __init__(
        self,
        name: tp.Optional[str],
        value: tp.Any,
        kind: tp.Type[tp.Any],
        module: tp.Optional["Tree"],
    ):
        self.name = name
        self.value = value
        self.kind = kind
        self.module = module


class TreeMeta(ABCMeta):
    def __call__(cls, *args, **kwargs) -> "Tree":
        obj: Tree = cls.__new__(cls)

        obj._field_metadata = obj._field_metadata.copy()

        if not dataclasses.is_dataclass(cls):
            for field, default_factory in obj._factory_fields.items():
                setattr(obj, field, default_factory())

            for field, default_value in obj._default_field_values.items():
                setattr(obj, field, default_value)

        obj.__init__(*args, **kwargs)

        # auto-annotations
        for field, value in vars(obj).items():

            if field not in obj._field_metadata and isinstance(value, Tree):
                obj._field_metadata[field] = types.FieldMetadata(
                    node=True,
                    kind=type(value),
                    opaque=False,
                    opaque_is_equal=None,
                )

        return obj


class Tree(types.KindMixin, metaclass=TreeMeta):
    _field_metadata: tp.Dict[str, types.FieldMetadata]
    _factory_fields: tp.Dict[str, tp.Callable[[], tp.Any]]
    _default_field_values: tp.Dict[str, tp.Any]

    @property
    def field_metadata(self) -> tp.Mapping[str, types.FieldMetadata]:
        return MappingProxyType(self._field_metadata)

    def update_field_metadata(
        self: T,
        field: str,
        node: tp.Optional[bool] = None,
        kind: tp.Optional[type] = None,
        opaque: tp.Optional[bool] = None,
        opaque_is_equal: tp.Union[
            tp.Callable[[utils.Opaque, tp.Any], bool],
            None,
            types.Missing,
        ] = types.MISSING,
    ) -> T:
        module = copy(self)

        field_metadata = module._field_metadata[field]

        if node is not None:
            field_metadata.node = node

        if kind is not None:
            field_metadata.kind = kind

        if opaque is not None:
            field_metadata.opaque = opaque

        if opaque_is_equal is not types.MISSING:
            field_metadata.opaque_is_equal = opaque_is_equal

        self._field_metadata[field] = field_metadata

        return module

    def __init_subclass__(cls):
        jax.tree_util.register_pytree_node_class(cls)

        # Restore the signature
        sig = inspect.signature(cls.__init__)
        parameters = tuple(sig.parameters.values())
        cls.__signature__ = sig.replace(parameters=parameters[1:])

        annotations = utils._get_all_annotations(cls)
        class_vars = utils._get_all_vars(cls)
        cls._field_metadata = {}
        cls._factory_fields = {}
        cls._default_field_values = {}

        for field, value in class_vars.items():
            if isinstance(value, dataclasses.Field):

                # save defaults
                if value.default is not dataclasses.MISSING:
                    cls._default_field_values[field] = value.default
                elif value.default_factory is not dataclasses.MISSING:
                    cls._factory_fields[field] = value.default_factory

                # extract metadata
                if value.metadata is not None and "node" in value.metadata:
                    cls._field_metadata[field] = types.FieldMetadata(
                        node=value.metadata["node"],
                        kind=value.metadata["kind"],
                        opaque=value.metadata["opaque"],
                        opaque_is_equal=value.metadata["opaque_is_equal"],
                    )

        for field, value in annotations.items():

            if field not in cls._field_metadata:
                is_node = any(issubclass(t, Tree) for t in utils._all_types(value))
                cls._field_metadata[field] = types.FieldMetadata(
                    node=is_node,
                    kind=type(None),
                    opaque=False,
                    opaque_is_equal=None,
                )

    def tree_flatten(self):

        fields = vars(self)

        node_fields = {}
        static_fields = {}

        if _CONTEXT.flatten_mode == FlattenMode.all_fields:
            node_fields = fields
        elif _CONTEXT.flatten_mode == FlattenMode.no_fields:
            static_fields = fields
        else:  # normal or None
            for field, value in fields.items():
                # maybe update metadata
                if field not in self._field_metadata:
                    is_node = isinstance(value, Tree)
                    self._field_metadata[field] = types.FieldMetadata(
                        node=is_node,
                        kind=type(None),
                        opaque=False,
                        opaque_is_equal=None,
                    )

                field_annotation = self._field_metadata[field]

                if field_annotation.node:
                    node_fields[field] = value
                elif not field_annotation.node and field_annotation.opaque:
                    static_fields[field] = utils.Opaque(
                        value,
                        opaque_is_equal=field_annotation.opaque_is_equal,
                    )
                else:
                    static_fields[field] = value

        # maybe convert to FieldInfo
        if _CONTEXT.add_field_info:
            for field in node_fields.keys():
                field_annotation = self._field_metadata[field]
                # leaves, treedef
                node_fields[field] = jax.tree_map(
                    lambda x: FieldInfo(
                        name=field,
                        value=x,
                        kind=field_annotation.kind,
                        module=self,
                    )
                    if not isinstance(x, Tree)
                    else x,
                    node_fields[field],
                    is_leaf=lambda x: isinstance(x, Tree),
                )

        children = (node_fields,)

        return children, static_fields

    @classmethod
    def tree_unflatten(cls, static_fields, children):

        module = cls.__new__(cls)
        (node_fields,) = children

        if _CONTEXT.add_field_info:
            for field in node_fields.keys():
                node_fields[field] = jax.tree_map(
                    lambda x: x.value if isinstance(x, FieldInfo) else x,
                    node_fields[field],
                    is_leaf=lambda x: isinstance(x, Tree),
                )

        module.__dict__.update(node_fields, **static_fields)

        # extract value from Opaque
        for field, value in static_fields.items():
            if (
                isinstance(value, utils.Opaque)
                and field in module._field_metadata
                and module._field_metadata[field].opaque
            ):
                setattr(module, field, value.value)

        return module


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
    Functional version of `Tree.filter` but can filter arbitrary pytrees. This is useful
    if you have Trees that are embedded in a larger pytree e.g. a list of Trees.

    Leaves that are not part of a Tree will get assigned the following `FieldInfo`:

    ```python
     FieldInfo(
        name=None,
        value=leaf_value,
        kind=type(None),
        module=None,
    )
    ```
    where `leaf_value` is the value of the leaf. This means that non-Tree leaves will be
    filtered with a query like:

    ```python
    tree = dict(a=1, b=to.Linear(3, 4))
    filtered = filter(tree, to.Parameter)

    assert isinstance(filtered["a"], to.Nothing)
    ```

    However, you query non-Tree based on their value:

    ```python
    tree = dict(a=1, b=to.Linear(3, 4))
    filtered = filter(tree, lambda field: isintance(field.value, int))

    assert filtered["a"] == 1
    ```

    If `inplace` is `True`, the input `obj` is mutated and returned.

    Arguments:
        obj: A pytree (possibly containing `to.Tree`s) to be filtered.
        filters: Types to filter by, membership is determined by `issubclass`, or
            callables that take in a `FieldInfo` and return a `bool`.
        inplace: If `True`, the input `obj` is mutated and returned.
    Returns:
        The new module with the filtered fields. If `inplace` is `True`, `obj` is returned.

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

    with _Context(add_field_info=True), _flatten_context(flatten_mode):
        obj = jax.tree_map(apply_filters, obj)

    if inplace:
        input_obj.__dict__.update(obj.__dict__)
        return input_obj
    else:
        return obj


def update(
    obj: A,
    other: A,
    *rest: A,
    inplace: bool = False,
    flatten_mode: tp.Union[FlattenMode, str, None] = None,
    ignore_static: bool = False,
) -> A:
    """
    Creates a new Tree with the same structure, but its values
    updated based on the values from the incoming Trees. Updates are performed using
    the following rules:

    * For a list of equivalent leaves `l1, l2, ..., ln`, it returns the first non-`Nothing` leaf in reverse order.
    * If no `flatten_mode()` context manager is active, and `flatten_mode` is not given, all fields will be updated.
    * If `flatten_mode="normal"` is set, the output will have the exact same static components as the first input (`obj`).

    Example:

    ```python
    t1 = MyTree(x=Nothing, y=2, z=3)
    t2 = MyTree(x=1, y=Nothing, z=4)

    update(t1, t2) # MyTree(x=1, y=2, z=4)
    ```

    When using `update` with multiple Trees the following equivalence holds:

    ```
    update(m1, m2, m3) = update(m1, update(m2, m3))
    ```

    If you want to update the current module instead of creating a new one use `inplace=True`.
    This is useful when applying transformation inside a method where reassigning `self` is not possible:

    ```python
    def double_params(self):
        # this is not doing what you expect
        self = jax.tree_map(lambda x: 2 * x, self)
    ```
    Instead do this:

    ```python
    def double_params(self):
        doubled = jax.tree_map(lambda x: 2 * x, self)
        update(self, doubled, inplace=True)
    ```

    If `inplace` is `True`, the input `obj` is mutated and returned. You can only update inplace if the input `obj` has a `__dict__` attribute, else a TypeError is raised.

    If `ignore_static` is `True`, static fields (according to the flattening mode) will be bypassed during the update process, the final output will have the same static components as the first input (`obj`). This strategy is a bit less safe in general as it will flatten all trees using `jax.tree_leaves` instead of `PyTreeDef.flatten_up_to`, this skips some checks so it effectively ignores their static components, the only requirement is that the flattened struture of all trees matches.

    Arguments:
        obj: Main pytree to update.
        other: The pytree first to get the values to update from.
        *rest: Additional pytree to perform the update in order from left to right.
        inplace: If `True`, the input `obj` is mutated and returned.
        static_from_first: If `True`, the static fields for the output are taken from the first input.

    Returns:
        A new pytree with the updated values. If `inplace` is `True`, `obj` is returned.
    """

    if inplace and not hasattr(obj, "__dict__"):
        raise TypeError(
            f"Cannot update inplace on objects with no __dict__ property, got {obj}"
        )

    if flatten_mode is None and _CONTEXT.flatten_mode is None:
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
) -> A:
    """
    Functional version of `Tree.map` but it can be applied to any pytree, useful if
    you have Trees that are embedded in a pytree. The `filters` are applied according
    to `to.filter`.

    Example:

    ```python
    tree = dict(a=1, b=MyModule().init(42))
    tree = to.map(jnp.zeros, tree, to.BatchStat)

    # "a" is not modified
    assert tree["a"] == 1
    ```

    Arguments:
        f: The function to apply to the leaves.
        obj: a pytree possibly containing `to.Tree`s.
        *filters: The filters used to select the leaves to which the function will be applied.
        inplace: If `True`, the input `obj` is mutated and returned.

    Returns:
        The object with the changes applied. If `inplace` is `True`, the input `obj` is returned.
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

        new_obj: A = jax.tree_map(f, new_obj)

        if has_filters:
            new_obj = update(obj, new_obj)

    if inplace:
        input_obj.__dict__.update(new_obj.__dict__)
        return input_obj
    else:
        return new_obj


def copy(obj: A) -> A:
    """
    Returns a deep copy of the tree, almost equivalent to:
    ```python
    jax.tree_map(lambda x: x, self)
    ```
    but Treeo will try to copy static nodes as well.
    """
    with _CONTEXT.update(flatten_mode=FlattenMode.all_fields):
        return jax.tree_map(lambda x: x, obj)


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
    rest = copy(rest)

    if not inplace:
        obj = copy(obj)

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


@contextmanager
def add_field_info():
    """
    A context manager that makes `Tree`s produce leaves as `FieldInfo` when flattening.
    """
    with _CONTEXT.update(add_field_info=True):
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

        with _CONTEXT.update(flatten_mode=mode):
            yield
    else:
        yield


# alias for internal use
_flatten_context = flatten_mode

# --------------------------------------------------
# utils
# --------------------------------------------------


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
