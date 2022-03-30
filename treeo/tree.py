import dataclasses
import enum
import functools
import inspect
import threading
import typing as tp
from abc import ABCMeta
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType

import jax
import jax.tree_util

from treeo import types, utils

T = tp.TypeVar("T", bound="Tree")
A = tp.TypeVar("A")


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


@dataclass
class _CompactContext(threading.local):
    current_tree: tp.Optional["Tree"] = None
    existing_subtrees: tp.Optional[tp.List["Tree"]] = None
    tree_idx: int = 0
    new_subtrees: tp.Optional[tp.List["Tree"]] = None
    compact_calls: tp.Optional[tp.Set[tp.Callable[..., tp.Any]]] = None

    @property
    def in_compact(self):
        return self.existing_subtrees is not None or self.new_subtrees is not None

    def __enter__(self):
        global _COMPACT_CONTEXT
        self._old_context = _COMPACT_CONTEXT
        _COMPACT_CONTEXT = self

    def __exit__(self, *args):
        global _COMPACT_CONTEXT
        _COMPACT_CONTEXT = self._old_context

    @contextmanager
    def update(self, **kwargs):
        fields = vars(self).copy()
        fields.pop("_old_context", None)
        fields.update(kwargs)

        with _CompactContext(**fields):
            yield

    @contextmanager
    def compact(self, f: tp.Callable[..., tp.Any], tree: "Tree"):
        new_subtrees: tp.Optional[tp.List["Tree"]]
        existing_subtrees: tp.Optional[tp.List["Tree"]]

        if tree._subtrees is None:
            existing_subtrees = None
            new_subtrees = []
        else:
            existing_subtrees = list(getattr(tree, field) for field in tree._subtrees)
            new_subtrees = None

        if self.current_tree is tree:
            assert self.compact_calls is not None

            if f in self.compact_calls:
                raise RuntimeError(f"Detected recursion in `compact` for: {tree}")

            self.compact_calls.add(f)
            yield
        else:
            with _CompactContext(
                current_tree=tree,
                existing_subtrees=existing_subtrees,
                new_subtrees=new_subtrees,
                compact_calls={f},
            ):
                yield

        if tree._subtrees is None:
            with _make_mutable_single(tree):
                assert new_subtrees is not None
                field_names = list(
                    utils._unique_names(
                        (utils._get_name(new_tree) for new_tree in new_subtrees),
                        existing_names=set(vars(tree).keys()),
                    )
                )
                tree._subtrees = tuple(field_names)
                for field, subtree in zip(field_names, new_subtrees):
                    if (
                        field in tree._field_metadata
                        and not tree._field_metadata[field].node
                    ):
                        raise ValueError(
                            f"Trying to subtree '{type(subtree).__name__}' to field '{field}' of '{type(tree).__name__}' but it has previously been declared with `node=False`"
                        )

                    if field not in tree._field_metadata:
                        tree._field_metadata[field] = types.FieldMetadata(
                            kind=type(None),
                            node=True,
                            opaque=False,
                        )

                    setattr(tree, field, subtree)


@dataclass
class _MutableContext(threading.local):
    prev_mutable: tp.Optional[tp.Dict["Tree", bool]] = None

    def __enter__(self):
        global _MUTABLE_CONTEXT
        self._old_context = _MUTABLE_CONTEXT
        _MUTABLE_CONTEXT = self

    def __exit__(self, *args):
        global _MUTABLE_CONTEXT
        _MUTABLE_CONTEXT = self._old_context

    @contextmanager
    def update(self, **kwargs):
        fields = vars(self).copy()
        fields.pop("_old_context", None)
        fields.update(kwargs)

        with _MutableContext(**fields):
            yield


_COMPACT_CONTEXT = _CompactContext()
_CONTEXT = _Context()
_MUTABLE_CONTEXT = _MutableContext()


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

    def __repr__(self) -> str:
        return f"FieldInfo(name={self.name!r}, value={self.value!r}, kind={self.kind!r}, module={type(self.module).__name__})"


class TreeMeta(ABCMeta):
    def __call__(cls: tp.Type[T], *args, **kwargs) -> T:
        obj: T

        if _COMPACT_CONTEXT.existing_subtrees is not None:
            if len(_COMPACT_CONTEXT.existing_subtrees) <= _COMPACT_CONTEXT.tree_idx:
                raise ValueError(
                    f"Out of bounds: trying to new initialize new Tree '{cls.__name__}' after the first compact run"
                )

            obj = _COMPACT_CONTEXT.existing_subtrees[_COMPACT_CONTEXT.tree_idx]
            _COMPACT_CONTEXT.tree_idx += 1

            return obj
        else:
            obj = cls.__new__(cls)
            with _make_mutable_single(obj):
                obj = cls.construct(obj, *args, **kwargs)

        if _COMPACT_CONTEXT.new_subtrees is not None:
            _COMPACT_CONTEXT.new_subtrees.append(obj)

        if _COMPACT_CONTEXT.in_compact and _COMPACT_CONTEXT.current_tree is not None:
            _set_mutable(obj, _COMPACT_CONTEXT.current_tree._mutable)

            if _MUTABLE_CONTEXT.prev_mutable is not None:
                _MUTABLE_CONTEXT.prev_mutable[obj] = False

        return obj

    def construct(cls, obj: T, *args, **kwargs) -> T:

        obj._field_metadata = obj._field_metadata.copy()

        # set default fields
        for field, default_factory in obj._factory_fields.items():
            setattr(obj, field, default_factory())

        for field, default_value in obj._default_field_values.items():
            setattr(obj, field, default_value)

        # reset context before __init__ and add obj as current tree
        with _CompactContext(current_tree=obj):
            obj.__init__(*args, **kwargs)

        # auto-annotations
        obj._update_local_metadata()

        return obj


class Tree(metaclass=TreeMeta):
    _field_metadata: tp.Dict[str, types.FieldMetadata]
    _factory_fields: tp.Dict[str, tp.Callable[[], tp.Any]]
    _default_field_values: tp.Dict[str, tp.Any]
    _subtrees: tp.Optional[tp.Tuple[str, ...]]
    _mutable: bool

    @property
    def field_metadata(self) -> tp.Mapping[str, types.FieldMetadata]:
        return MappingProxyType(self._field_metadata)

    def update_field_metadata(
        self: T,
        field: str,
        node: tp.Optional[bool] = None,
        kind: tp.Optional[type] = None,
        opaque: tp.Union[bool, utils.OpaquePredicate, None] = None,
    ) -> T:
        module = copy(self)

        field_metadata = module._field_metadata[field]
        updates = {}

        if node is not None:
            updates.update(node=node)

        if kind is not None:
            updates.update(kind=kind)

        if opaque is not None:
            updates.update(opaque=opaque)

        if updates:
            field_metadata = field_metadata.update(**updates)

        module._field_metadata[field] = field_metadata

        return module

    def check_metadata_updates(self):
        """
        Checks for new fields, if found, adds them to the metadata.
        """
        with _CONTEXT.update(flatten_mode=FlattenMode.all_fields):
            jax.tree_flatten(self)

    def _update_local_metadata(self):
        for field, value in vars(self).items():

            if field not in self._field_metadata:
                self._field_metadata[field] = types.FieldMetadata(
                    node=isinstance(value, Tree),
                    kind=type(value),
                    opaque=False,
                )

    def __init_subclass__(cls):
        jax.tree_util.register_pytree_node_class(cls)

        # Restore the signature
        sig = inspect.signature(cls.__init__)
        parameters = tuple(sig.parameters.values())
        cls.__signature__ = sig.replace(parameters=parameters[1:])

        annotations = utils._get_all_annotations(cls)
        class_vars = utils._get_all_vars(cls)

        # init class variables
        cls._field_metadata = {}
        cls._factory_fields = {}
        cls._default_field_values = {}
        cls._subtrees = None
        cls._mutable = False

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
                    )

        for field, value in annotations.items():

            if field not in cls._field_metadata:
                is_node = any(issubclass(t, Tree) for t in utils._all_types(value))
                cls._field_metadata[field] = types.FieldMetadata(
                    node=is_node,
                    kind=type(None),
                    opaque=False,
                )

    def tree_flatten(self):

        fields = vars(self).copy()

        node_fields = {}
        static_fields = {}

        # auto-annotations
        self._update_local_metadata()

        if _CONTEXT.flatten_mode == FlattenMode.all_fields:
            node_fields = fields
        elif _CONTEXT.flatten_mode == FlattenMode.no_fields:
            static_fields = fields
        else:  # normal or None
            for field, value in fields.items():
                field_annotation = self._field_metadata[field]

                if field_annotation.node:
                    node_fields[field] = value
                elif not field_annotation.node and field_annotation.opaque != False:
                    static_fields[field] = utils.Opaque(
                        value,
                        predicate=field_annotation.opaque
                        if not isinstance(field_annotation.opaque, bool)
                        else None,
                    )
                else:
                    static_fields[field] = value

        # maybe convert to FieldInfo
        if _CONTEXT.add_field_info:
            for field in node_fields.keys():
                if field in TREE_PRIVATE_FIELDS:
                    continue

                kind = self._field_metadata[field].kind
                # leaves, treedef
                node_fields[field] = jax.tree_map(
                    lambda x: FieldInfo(
                        name=field,
                        value=x,
                        kind=kind,
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

        with _make_mutable_single(module):
            # extract value from Opaque
            for field, value in static_fields.items():
                if (
                    isinstance(value, utils.Opaque)
                    and field in module._field_metadata
                    and module._field_metadata[field].opaque
                ):
                    setattr(module, field, value.value)

        return module


TREE_PRIVATE_FIELDS = {
    x for x in list(vars(Tree)) + list(Tree.__annotations__) if x.startswith("_")
}


def copy(obj: A) -> A:
    """
    Returns a deep copy of the tree, almost equivalent to:
    ```python
    jax.tree_map(lambda x: x, self)
    ```
    but will try to copy static nodes as well.
    """
    with _CONTEXT.update(flatten_mode=FlattenMode.all_fields):
        return jax.tree_map(lambda x: x, obj)


def apply(
    f: tp.Callable[..., None],
    obj: A,
    *rest: A,
    inplace: bool = False,
    _top_inplace: tp.Optional[bool] = None,
) -> A:
    """
    Applies a function to all `to.Tree`s in a Pytree. Works very similar to `jax.tree_map`,
    but its values are `to.Tree`s instead of leaves, also `f` should apply the changes inplace to Tree object.

    If `inplace` is `False`, a copy of the first object is returned with the changes applied.
    The `rest` of the objects are always copied.

    If a `Tree` is `Immutable` and `inplace=True` a `RuntimeError` could be raised if a field is mutated.

    Arguments:
        f: The function to apply.
        obj: a pytree possibly containing Trees.
        *rest: additional pytrees.
        inplace: If `True`, the input `obj` is mutated.

    Returns:
        A new pytree with the updated Trees or the same input `obj` if `inplace` is `True`.
    """
    if _top_inplace is None:
        _top_inplace = inplace

    rest = copy(rest)
    if not inplace:
        obj = copy(obj)

    objs = (obj,) + rest

    def nested_fn(obj, *rest):
        if isinstance(obj, Tree):
            apply(f, obj, *rest, inplace=True, _top_inplace=_top_inplace)

    jax.tree_map(
        nested_fn,
        *objs,
        is_leaf=lambda x: isinstance(x, Tree) and not x in objs,
    )

    if isinstance(obj, Tree):
        if _top_inplace:
            f(obj, *rest)
        else:
            with _make_mutable_single(obj):
                f(obj, *rest)

    return obj


@contextmanager
def _make_mutable(obj: tp.Any):
    """
    Context manager that makes the tree mutable.
    """

    def _make_mutable_fn(a: Tree):
        assert _MUTABLE_CONTEXT.prev_mutable is not None
        _MUTABLE_CONTEXT.prev_mutable[a] = a._mutable
        # update __dict__ instead to avoid error during __setattr__
        a.__dict__["_mutable"] = True

    def _revert_mutable_fn(a: Tree):
        assert _MUTABLE_CONTEXT.prev_mutable is not None
        # update __dict__ instead to avoid error during __setattr__
        a.__dict__["_mutable"] = _MUTABLE_CONTEXT.prev_mutable[a]

    with _MUTABLE_CONTEXT.update(prev_mutable={}):
        try:
            apply(_make_mutable_fn, obj, inplace=True)
            yield
        finally:
            apply(_revert_mutable_fn, obj, inplace=True)


@contextmanager
def _make_mutable_single(*objs: Tree):
    """
    Context manager that makes a single tree mutable.
    """

    _mutables = [obj._mutable for obj in objs]

    for obj in objs:
        # update __dict__ instead to avoid error during __setattr__
        obj.__dict__["_mutable"] = True

    try:
        yield
    finally:
        # update __dict__ instead to avoid error during __setattr__
        for obj, _mutable in zip(objs, _mutables):
            obj.__dict__["_mutable"] = _mutable


def _set_mutable(obj: Tree, value: bool):
    obj.__dict__["_mutable"] = value
