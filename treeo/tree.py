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


class _FlattenMode(enum.Enum):
    normal = enum.auto()
    all_static = enum.auto()
    all_dynamic = enum.auto()


@dataclass
class _Context(threading.local):
    add_field_info: bool = False
    flatten_mode: _FlattenMode = _FlattenMode.normal

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
    _init_called: bool = False
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
        module = self.copy()
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
            if field not in cls._field_metadata and any(
                issubclass(t, Tree) for t in utils._all_types(value)
            ):
                cls._field_metadata[field] = types.FieldMetadata(
                    node=True,
                    kind=value,
                    opaque=False,
                    opaque_is_equal=None,
                )

    def tree_flatten(self):

        fields = vars(self)

        tree = {}
        not_tree = {}

        if _CONTEXT.flatten_mode == _FlattenMode.all_dynamic:
            tree = fields
        elif _CONTEXT.flatten_mode == _FlattenMode.all_static:
            not_tree = fields
        else:
            for field, value in fields.items():
                # maybe update metadata
                if field not in self._field_metadata and isinstance(value, Tree):
                    self._field_metadata[field] = types.FieldMetadata(
                        node=True,
                        kind=type(value),
                        opaque=False,
                        opaque_is_equal=None,
                    )

                field_annotation = self._field_metadata.get(field, None)

                if field_annotation is not None:
                    if field_annotation.node:
                        tree[field] = value
                    elif not field_annotation.node and field_annotation.opaque:
                        not_tree[field] = utils.Opaque(
                            value,
                            opaque_is_equal=field_annotation.opaque_is_equal,
                        )
                    else:
                        not_tree[field] = value

                else:
                    not_tree[field] = value

        # maybe convert to FieldInfo
        if _CONTEXT.add_field_info:
            for field, value in tree.items():
                field_annotation = self._field_metadata[field]
                # leaves, treedef
                tree[field], not_tree[field] = jax.tree_flatten(value)
                tree[field] = [
                    FieldInfo(
                        name=field,
                        value=x,
                        kind=field_annotation.kind,
                        module=self,
                    )
                    if not isinstance(x, FieldInfo)
                    else x
                    for x in tree[field]
                ]

        children = (tree,)

        return children, not_tree

    @classmethod
    def tree_unflatten(cls, not_tree, children):

        module = cls.__new__(cls)
        (tree,) = children

        if _CONTEXT.add_field_info:
            for field, leaves in tree.items():
                treedef = not_tree.pop(field)
                tree[field] = jax.tree_unflatten(treedef, leaves)

        module.__dict__.update(tree, **not_tree)

        # extract value from Opaque
        for field, value in not_tree.items():
            if (
                isinstance(value, utils.Opaque)
                and field in module._field_metadata
                and module._field_metadata[field].opaque
            ):
                setattr(module, field, value.value)

        return module

    def copy(self: T) -> T:
        """
        Returns a deep copy of the module, implemented as:
        ```python
        jax.tree_map(lambda x: x, self)
        ```
        """
        with _CONTEXT.update(flatten_mode=_FlattenMode.all_dynamic):
            return jax.tree_map(lambda x: x, self)


# --------------------------------------------------
# functions
# --------------------------------------------------


def apply(f: tp.Callable[..., None], obj: A, *rest: A, inplace: bool = False) -> A:
    """
    Applies a function to all `to.Tree`s in a Pytree. Function very similar to `jax.tree_map`,
    but works on Trees instead of values and `f` should apply the changes inplace to the
    first object.

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
    rest = jax.tree_map(lambda x: x, rest)

    if not inplace:
        obj = jax.tree_map(lambda x: x, obj)

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


def map(f: tp.Callable, obj: A, *filters: Filter) -> A:
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
        filters: The filters used to select the leaves to which the function will be applied.

    Returns:
        The object with the changes applied.
    """

    has_filters = len(filters) > 0

    if has_filters:
        new_obj = filter(obj, *filters)
    else:
        new_obj = obj

    new_obj: A = jax.tree_map(f, new_obj)

    if has_filters:
        new_obj = update(obj, new_obj)

    return new_obj


def filter(obj: A, *filters: Filter) -> A:
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

    Arguments:
        filters: Types to filter by, membership is determined by `issubclass`, or
            callables that take in a `FieldInfo` and return a `bool`.
    Returns:
        The new module with the filtered fields.

    """

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

    with _Context(add_field_info=True):
        obj = jax.tree_map(apply_filters, obj)

    return obj


def update(module: A, other: A, *rest: A) -> A:
    """
    Functional version of `Module.update`, it accepts arbitray pytree structures
    that may optionally contain `Module`s and performs the `update` logic.

    Arguments:
        module: Main pytree to update.
        other: The pytree first to get the values to update from.
        rest: Additional pytree to perform the update in order from left to right.

    Returns:
        A new pytree with the updated values.
    """

    def merge_fn(*xs):
        acc, *xs = xs
        for x in xs:
            if not isinstance(x, types.Nothing):
                acc = x
        return acc

    module = _looser_tree_map(
        merge_fn,
        module,
        other,
        *rest,
        is_leaf=lambda x: isinstance(x, LEAF_TYPES),
    )

    return module


@contextmanager
def add_field_info():
    """
    A context manager that makes `Tree`s return leaves as `FieldInfo` when flattening.
    """
    with _CONTEXT.update(add_field_info=True):
        yield


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
