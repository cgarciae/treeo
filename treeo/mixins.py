import dataclasses
import functools
import inspect
import threading
import typing as tp
from contextlib import contextmanager

from treeo import api
from treeo import tree as tree_m
from treeo import types, utils

A = tp.TypeVar("A")
C = tp.TypeVar("C", bound=tp.Callable)


class Copy:
    """
    Mixin that adds a `.copy()` method to the class.
    """

    def copy(self: A) -> A:
        """
        `copy` is a wrapper over `treeo.copy` that passes `self` as the first argument.
        """
        return tree_m.copy(self)


class ToString:
    """
    Mixin that adds a `.to_string()` method to the class.
    """

    def to_string(
        self: A,
        *,
        private_fields: bool = False,
        static_fields: bool = True,
        color: bool = False,
    ) -> str:
        """
        `to_string` is a wrapper over `treeo.to_string` that passes `self` as the first argument.

        Arguments:
            private_fields: If `True`, private fields are included.
            static_fields: If `True`, static fields are included.
            color: If `True`, color is included.

        Returns:
            A string representation of the object.
        """
        return api.to_string(
            self,
            private_fields=private_fields,
            static_fields=static_fields,
            color=color,
        )


class ToDict:
    """
    Mixin that adds a `.to_dict()` method to the class.
    """

    def to_dict(
        self,
        *,
        private_fields: bool = False,
        static_fields: bool = True,
        type_info: bool = False,
        field_info: bool = False,
    ) -> tp.Any:
        """
        `to_dict` is a wrapper over `treeo.to_dict` that passes `self` as the first argument.

        Arguments:
            private_fields: If `True`, private fields are included.
            static_fields: If `True`, static fields are included.
            type_info: If `True`, type information is included.
            field_info: If `True`, field information is included.

        Returns:
            A dict representation of the object.
        """
        return api.to_dict(
            self,
            private_fields=private_fields,
            static_fields=static_fields,
            type_info=type_info,
            field_info=field_info,
        )


class Repr:
    """
    Mixin that adds a `__repr__` method to the class.
    """

    def __repr__(self) -> tp.Any:
        """
        Uses `treeo.to_string` to generate a string representation of the object.
        """
        return api.to_string(
            self,
            private_fields=False,
            static_fields=True,
            color=False,
        )


class Filter:
    """
    Mixin that adds a `.filter()` method to the class.
    """

    def filter(
        self: A,
        *filters: api.Filter,
        flatten_mode: tp.Union[api.FlattenMode, str, None] = None,
    ) -> A:
        """
        `filter` is a wrapper over `treeo.filter` that passes `self` as the first argument.

        Arguments:
            *filters: Types to filter by, membership is determined by `issubclass`, or
                callables that take in a `FieldInfo` and return a `bool`.
            flatten_mode: Sets a new `FlattenMode` context for the operation.

        Returns:
            A new pytree with the filtered fields.
        """
        return api.filter(self, *filters, flatten_mode=flatten_mode)


class Merge:
    """
    Mixin that adds a `.merge()` method to the class.
    """

    def merge(
        self: A,
        other: A,
        *rest: A,
        flatten_mode: tp.Union[api.FlattenMode, str, None] = None,
        ignore_static: bool = False,
    ) -> A:
        """
        `merge` is a wrapper over `treeo.merge` that passes `self` as the first argument.

        Arguments:
            other: The pytree first to get the values to merge with.
            *rest: Additional pytree to perform the merge in order from left to right.
            flatten_mode: Sets a new `FlattenMode` context for the operation, if `None` the current context is used. If the current flatten context is `None` and `flatten_mode` is not passed then `FlattenMode.all_fields` is used.
            ignore_static: If `True`, bypasses static fields during the process and the statics fields for output are taken from the first input (`obj`).

        Returns:
            A new pytree with the merged values.
        """
        return api.merge(
            self,
            other,
            *rest,
            flatten_mode=flatten_mode,
            ignore_static=ignore_static,
        )


class Map:
    """
    Mixin that adds a `.map()` method to the class.
    """

    def map(
        self: A,
        f: tp.Callable,
        *filters: api.Filter,
        flatten_mode: tp.Union[api.FlattenMode, str, None] = None,
        is_leaf: tp.Callable[[tp.Any], bool] = None,
    ) -> A:
        """
        `map` is a wrapper over `treeo.map` that passes `self` as the second argument.

        Arguments:
            f: The function to apply to the leaves.
            *filters: The filters used to select the leaves to which the function will be applied.
            flatten_mode: Sets a new `FlattenMode` context for the operation, if `None` the current context is used.

        Returns:
            A new pytree with the changes applied.
        """
        return api.map(
            f,
            self,
            *filters,
            flatten_mode=flatten_mode,
            is_leaf=is_leaf,
        )


class Apply:
    """
    Mixin that adds a `.apply()` method to the class.
    """

    def apply(self: A, f: tp.Callable[..., None], *rest: A, inplace: bool = False) -> A:
        """
        `apply` is a wrapper over `treeo.apply` that passes `self` as the second argument.

        Arguments:
            f: The function to apply.
            *rest: additional pytrees.
            inplace: If `True`, the input `obj` is mutated.

        Returns:
            A new pytree with the updated Trees or the same input `obj` if `inplace` is `True`.
        """
        return tree_m.apply(f, self, *rest, inplace=inplace)


class Compact:
    _field_metadata: tp.Dict[str, types.FieldMetadata]
    _subtrees: tp.Optional[tp.Tuple[str, ...]]

    @property
    def first_run(self) -> bool:
        """
        Returns:
            `True` if its currently the first run of a `compact` method.
        """
        if tree_m._COMPACT_CONTEXT.current_tree is not self:
            raise RuntimeError(
                f"Object '{type(self).__name__}' is not the current tree, found '{type(tree_m._COMPACT_CONTEXT.current_tree).__name__}', did you forget the @compact decorator?"
            )

        return self._subtrees is None

    # NOTE: it feels like `get_field` could be safely used in non-`compact` methods, maybe
    # the various checks done to verify that this method is used inside `compact` could be removed.
    def get_field(
        self,
        field_name: str,
        initializer: tp.Callable[[], A],
    ) -> A:
        """
        A method that gets a field with the given name if exists, otherwise it initializes it and returns it.

        Currently the follow restrictions apply:

        * The field must be declared in the class definition.
        * The method can only be called inside a `compact` context.

        Arguments:
            field_name: The name of the field to get.
            initializer: The function to initialize the field if it does not exist.

        Returns:
            The field value.
        """
        value: A

        if field_name not in self._field_metadata:
            raise ValueError(f"Metadata for field '{field_name}' does not exist.")

        if field_name in vars(self):
            value = getattr(self, field_name)
        else:
            if tree_m._COMPACT_CONTEXT.in_compact and not self.first_run:
                raise RuntimeError(
                    f"Trying to initialize field '{field_name}' after the first run of `compact`."
                )

            value = initializer()

            with tree_m._make_mutable_toplevel(self):
                setattr(self, field_name, value)

        return value


class Extensions(Copy, ToString, ToDict, Repr, Filter, Merge, Map, Apply, Compact):
    """
    Mixin that adds all available mixins from `treeo.mixins` except `KindMixin`.
    """

    pass


class KindMixin:
    @classmethod
    def field(
        cls,
        default: tp.Any = dataclasses.MISSING,
        *,
        node: bool,
        default_factory: tp.Any = dataclasses.MISSING,
        init: bool = True,
        repr: bool = True,
        hash: tp.Optional[bool] = None,
        compare: bool = True,
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
        )

    @classmethod
    def node(
        cls,
        default: tp.Any = dataclasses.MISSING,
        *,
        default_factory: tp.Any = dataclasses.MISSING,
        init: bool = True,
        repr: bool = True,
        hash: tp.Optional[bool] = None,
        compare: bool = True,
    ) -> tp.Any:
        return utils.node(
            default=default,
            kind=cls,
            default_factory=default_factory,
            init=init,
            repr=repr,
            hash=hash,
            compare=compare,
        )

    @classmethod
    def static(
        cls,
        default: tp.Any = dataclasses.MISSING,
        default_factory: tp.Any = dataclasses.MISSING,
        init: bool = True,
        repr: bool = True,
        hash: tp.Optional[bool] = None,
        compare: bool = True,
    ) -> tp.Any:
        return cls.field(
            default=default,
            node=False,
            default_factory=default_factory,
            init=init,
            repr=repr,
            hash=hash,
            compare=compare,
        )


# -------------------------------------------------------------------------------
# Immutable Mixin
# -------------------------------------------------------------------------------


class Immutable:
    """
    Mixin that makes a class immutable. It adds a `.replace()` and `.mutable()` methods to the class
    which let you modify the state by creating a new objects.
    """

    _mutable: tp.Optional[tree_m._MutableState]

    @property
    def is_mutable(self) -> bool:
        """
        Returns:
            `True` if the object is mutable.
        """
        return self._mutable is not None

    def replace(self: A, **kwargs) -> A:
        """
        Returns a copy of the Tree with the given fields specified in `kwargs`
        updated to the new values.

        Example:

        ```python
        @dataclass
        class MyTree(to.Tree, to.Immutable):
            x: int = to.node()

        tree = MyTree(x=1)

        # increment x by 1
        tree = tree.replace(x=tree.x + 1)
        ```

        Arguments:
            **kwargs: The fields to update.

        Returns:
            A new Tree with the updated fields.
        """
        tree: tree_m.Tree = tree_m.copy(self)

        with tree_m._make_mutable_toplevel(tree):
            for key, value in kwargs.items():
                setattr(tree, key, value)

        tree._update_local_metadata()
        # return a copy to potentially update metadata
        return tree


class MutabilityError(AttributeError):
    """
    Raised when an operation is attempted on an immutable object.
    """

    pass


# define __setattr__ outside of class so linters still detect it unknown attribute assignments
def _immutable_setattr(self: Immutable, key: str, value: tp.Any) -> None:
    if not self._mutable:
        raise MutabilityError(
            f"Trying to mutate field '{key}' in immutable '{type(self).__name__}' object."
        )

    object.__setattr__(self, key, value)


Immutable.__setattr__ = _immutable_setattr


class ImmutableTree(tree_m.Tree, Immutable):
    """
    A Tree class that also inherits from `Immutable`.
    """

    pass
