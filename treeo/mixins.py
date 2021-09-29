import typing as tp

from treeo import api

A = tp.TypeVar("A")


class Copy:
    """
    Mixin that adds a `.copy()` method to the class.
    """

    def copy(self):
        """
        `copy` is a wrapper over `treeo.copy` that passes `self` as the first argument.
        """
        return api.copy(self)


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
            other: The pytree first to get the values to update from.
            *rest: Additional pytree to perform the update in order from left to right.
            inplace: If `True`, the input `obj` is mutated and returned.
            flatten_mode: Sets a new `FlattenMode` context for the operation, if `None` the current context is used. If the current flatten context is `None` and `flatten_mode` is not passed then `FlattenMode.all_fields` is used.
            ignore_static: If `True`, bypasses static fields during the process and the statics fields for output are taken from the first input (`obj`).

        Returns:
            A new pytree with the updated values. If `inplace` is `True`, `obj` is returned.
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
        return api.map(f, self, *filters, inplace=inplace, flatten_mode=flatten_mode)


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


class Extensions(Copy, ToString, ToDict, Repr, Filter, Merge, Map, Apply):
    """
    Mixin that adds all available mixins from `treeo.mixins`.
    """

    pass
