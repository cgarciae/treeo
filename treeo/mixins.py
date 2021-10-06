import dataclasses
import typing as tp

from treeo import api
from treeo import tree as tree_m
from treeo import types, utils

A = tp.TypeVar("A")


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
        inplace: bool = False,
        flatten_mode: tp.Union[api.FlattenMode, str, None] = None,
    ) -> A:
        """
        `filter` is a wrapper over `treeo.filter` that passes `self` as the first argument.

        Arguments:
            *filters: Types to filter by, membership is determined by `issubclass`, or
                callables that take in a `FieldInfo` and return a `bool`.
            inplace: If `True`, the input `obj` is mutated and returned.
            flatten_mode: Sets a new `FlattenMode` context for the operation.

        Returns:
            A new pytree with the filtered fields. If `inplace` is `True`, `obj` is returned.
        """
        return api.filter(self, *filters, inplace=inplace, flatten_mode=flatten_mode)


class Merge:
    """
    Mixin that adds a `.merge()` method to the class.
    """

    def merge(
        self: A,
        other: A,
        *rest: A,
        inplace: bool = False,
        flatten_mode: tp.Union[api.FlattenMode, str, None] = None,
        ignore_static: bool = False,
    ) -> A:
        """
        `merge` is a wrapper over `treeo.merge` that passes `self` as the first argument.

        Arguments:
            other: The pytree first to get the values to merge with.
            *rest: Additional pytree to perform the merge in order from left to right.
            inplace: If `True`, the input `obj` is mutated and returned.
            flatten_mode: Sets a new `FlattenMode` context for the operation, if `None` the current context is used. If the current flatten context is `None` and `flatten_mode` is not passed then `FlattenMode.all_fields` is used.
            ignore_static: If `True`, bypasses static fields during the process and the statics fields for output are taken from the first input (`obj`).

        Returns:
            A new pytree with the merged values. If `inplace` is `True`, `obj` is returned.
        """
        return api.merge(
            self,
            other,
            *rest,
            inplace=inplace,
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
        *filters: Filter,
        inplace: bool = False,
        flatten_mode: tp.Union[api.FlattenMode, str, None] = None,
        is_leaf: tp.Callable[[tp.Any], bool] = None,
    ) -> A:
        """
        `map` is a wrapper over `treeo.map` that passes `self` as the second argument.

        Arguments:
            f: The function to apply to the leaves.
            *filters: The filters used to select the leaves to which the function will be applied.
            inplace: If `True`, the input `obj` is mutated and returned.
            flatten_mode: Sets a new `FlattenMode` context for the operation, if `None` the current context is used.

        Returns:
            A new pytree with the changes applied. If `inplace` is `True`, the input `obj` is returned.
        """
        return api.map(
            f,
            self,
            *filters,
            inplace=inplace,
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
        return api.apply(f, self, *rest, inplace=inplace)


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

        if tree_m._COMPACT_CONTEXT.current_tree is not self:
            raise RuntimeError(
                f"Object '{type(self).__name__}' is not the current tree, found '{type(tree_m._COMPACT_CONTEXT.current_tree).__name__}', did you forget the @compact decorator?"
            )

        if field_name in vars(self):
            value = getattr(self, field_name)
        else:
            if tree_m._COMPACT_CONTEXT.in_compact and not self.first_run:
                raise RuntimeError(
                    f"Trying to initialize field '{field_name}' after the first run of `compact`."
                )

            value = initializer()
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
        default=dataclasses.MISSING,
        *,
        node: bool,
        default_factory=dataclasses.MISSING,
        init: bool = True,
        repr: bool = True,
        hash: tp.Optional[bool] = None,
        compare: bool = True,
        opaque: tp.Union[bool, utils.OpaquePredicate] = False,
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
            opaque=opaque,
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
        opaque: tp.Union[bool, utils.OpaquePredicate] = False,
    ) -> tp.Any:
        return utils.node(
            default=default,
            kind=cls,
            default_factory=default_factory,
            init=init,
            repr=repr,
            hash=hash,
            compare=compare,
            opaque=opaque,
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
        opaque: tp.Union[bool, utils.OpaquePredicate] = False,
    ) -> tp.Any:
        return cls.field(
            default=default,
            node=False,
            default_factory=default_factory,
            init=init,
            repr=repr,
            hash=hash,
            compare=compare,
            opaque=opaque,
        )
