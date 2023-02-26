import typing as tp
from dataclasses import dataclass
from inspect import Parameter

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
import pytest

import treeo as to
from treeo.api import T, add_field_info, map
from treeo.tree import Tree
from treeo.utils import field


class Parameter(to.KindMixin):
    pass


class State(to.KindMixin):
    pass


class Linear(to.Tree, to.Immutable):
    w: np.ndarray = Parameter.node()
    b: np.ndarray = Parameter.node()
    n: int = State.node()

    def __init__(self, din, dout, name="linear"):
        self.din = din
        self.dout = dout
        self.w = np.random.uniform(size=(din, dout))
        self.b = np.random.uniform(size=(dout,))
        self.n = 1
        self.name = name

    def __call__(self, x):
        self.n += 1
        return jnp.dot(x, self.w) + self.b

    def double(self, x):
        return self(x) * 2.0


class MLP(to.Tree, to.Immutable):
    linear1: Linear
    linear2: Linear

    def __init__(self, din, dmid, dout, name="mlp"):
        self.din = din
        self.dmid = dmid
        self.dout = dout
        self.name = name

        self.linear1 = Linear(din, dmid)
        self.linear2 = Linear(dmid, dout)

    def __call__(self, x):
        return self.linear2(self.linear1(x))


class TestImmutable:
    def test_mutable_error(self):
        x = np.random.uniform(size=(5, 2))
        linear = Linear(2, 3)

        with pytest.raises(to.MutabilityError):
            y = linear(x)

    def test_mutable_functional(self):
        x = np.random.uniform(size=(5, 2))
        linear = Linear(2, 3)

        y, linear2 = to.mutable(Linear.__call__)(linear, x)

        assert y.shape == (5, 3)
        assert isinstance(linear2, Linear)
        assert linear is not linear2

    def test_mutable_callable(self):
        x = np.random.uniform(size=(5, 2))
        linear = Linear(2, 3)

        y, linear2 = to.mutable(linear)(x)

        assert y.shape == (5, 3)
        assert isinstance(linear2, Linear)
        assert linear is not linear2

    def test_mutable_bounded_method(self):
        x = np.random.uniform(size=(5, 2))
        linear = Linear(2, 3)

        y, linear2 = to.mutable(linear.__call__)(x)

        assert y.shape == (5, 3)
        assert isinstance(linear2, Linear)
        assert linear is not linear2

    def test_mutable_unbounded_method(self):
        x = np.random.uniform(size=(5, 2))
        linear = Linear(2, 3)

        y, linear2 = to.mutable(Linear.__call__)(linear, x)

        assert y.shape == (5, 3)
        assert isinstance(linear2, Linear)
        assert linear is not linear2

    def test_mutable_string_method(self):
        x = np.random.uniform(size=(5, 2))
        linear = Linear(2, 3)

        y, linear2 = to.mutable(linear.double)(x)

        assert y.shape == (5, 3)
        assert isinstance(linear2, Linear)
        assert linear is not linear2

    def test_mutable_submodule(self):
        class Parent(to.Tree, to.Immutable):
            def __init__(self):
                self.child = Child()

            def __call__(self, x):
                return self.child(x)

        class Child(to.Tree, to.Immutable):
            n: int = to.node()

            def __init__(self):
                self.n = 0

            def __call__(self, x):
                self.n += 1
                return x

        x = np.random.uniform(size=(5, 2))
        module = Parent()

        with pytest.raises(to.MutabilityError):
            y = module(x)

        assert module.child.n == 0

        y, module2 = to.mutable(module)(x)

        assert not module._mutable
        assert not module.child._mutable
        assert not module2._mutable
        assert not module2.child._mutable

        assert y.shape == (5, 2)
        assert isinstance(module2, Parent)
        assert module is not module2
        assert module2.child.n == 1

        with pytest.raises(to.MutabilityError):
            to.mutable(module, toplevel_only=True)(x)

    def test_mutable_compact_submodule(self):
        class Parent(to.Tree, to.Immutable):
            @to.compact
            def __call__(self, x):
                return Child()(x)

        class Child(to.Tree, to.Immutable):
            n: int = to.node()

            def __init__(self):
                self.n = 0

            def __call__(self, x):
                self.n += 1
                return x

        x = np.random.uniform(size=(5, 2))
        module = Parent()

        y, module2 = to.mutable(module)(x)

        assert not module._mutable
        assert not module2._mutable
        assert not module2.child._mutable

        assert y.shape == (5, 2)
        assert isinstance(module2, Parent)
        assert module is not module2
        assert module2.child.n == 1

        with pytest.raises(to.MutabilityError):
            to.mutable(module, toplevel_only=True)(x)

    def test_mutable_compact_docorators(self):
        class Parent(to.Tree, to.Immutable):
            @to.mutable
            @to.compact
            def __call__(self, x):
                return Child()(x)

        class Child(to.Tree, to.Immutable):
            n: int = to.node()

            def __init__(self):
                self.n = 0

            def __call__(self, x):
                self.n += 1
                return x

        x = np.random.uniform(size=(5, 2))
        module = Parent()

        y, module2 = module(x)

        assert not module._mutable
        assert not module2._mutable
        assert not module2.child._mutable

        assert y.shape == (5, 2)
        assert isinstance(module2, Parent)
        assert module is not module2
        assert module2.child.n == 1

    def test_mutable_compact_docorators_invalid_order(self):
        class Child(to.Tree, to.Immutable):
            n: int = to.node()

            def __init__(self):
                self.n = 0

            def __call__(self, x):
                self.n += 1
                return x

        with pytest.raises(ValueError):

            class Parent(to.Tree, to.Immutable):
                @to.compact
                @to.mutable
                def __call__(self, x):
                    return Child()(x)

    def test_mutable_replace(self):
        class Parent(to.Tree, to.Immutable):
            def __init__(self):
                self.child = Child()

            def __call__(self, x):
                return self.child(x)

        class Child(to.Tree, to.Immutable):
            n: int = to.node()

            def __init__(self):
                self.n = 0

            def __call__(self, x):
                self.n += 1
                return x

        x = np.random.uniform(size=(5, 2))
        module = Parent()

        with to.make_mutable(module):
            module.child = Child()

    def test_mutable_returns_state(self):
        class Parent(to.Tree, to.Immutable):
            def __init__(self) -> None:
                self.child = Child()

            @to.compact
            def __call__(self, x, try_mutable_child: bool):
                if try_mutable_child:
                    x = self.child(x)
                else:
                    x, self.child = to.mutable(self.child)(x)
                return x, self

        class Child(to.Tree, to.Immutable):
            n: int = to.node()

            def __init__(self):
                self.n = 0

            def __call__(self, x):
                self.n += 1
                return x

        x = np.random.uniform(size=(5, 2))
        module = Parent()

        # with pytest.raises(RuntimeError):
        #     y = module(x)

        # assert not hasattr(module, "child")

        y, module2 = to.toplevel_mutable(module)(x, False)

        assert not module._mutable
        assert not module2._mutable
        assert not module2.child._mutable

        assert y.shape == (5, 2)
        assert isinstance(module2, Parent)
        assert module is not module2
        assert module2.child.n == 1

        with pytest.raises(to.MutabilityError):
            to.toplevel_mutable(module)(x, True)

        @to.toplevel_mutable
        def nested_mutable(module: Parent):
            assert module._mutable

            @to.toplevel_mutable
            def nested_fn(module: Parent):
                y, module2 = to.toplevel_mutable(module)(x, False)
                return module2, module

            module2, module = nested_fn(module)

            assert module._mutable
            assert not module2._mutable

            return module

        module = nested_mutable(module)
        assert not module._mutable

    def test_toplevel_mutable_nested(self):
        class Parent(to.Tree, to.Immutable):
            def __init__(self) -> None:
                self.child = Child()

            @to.compact
            def __call__(self, x, try_mutable_child: bool):
                if try_mutable_child:
                    x = self.child(x)
                else:
                    x, self.child = to.mutable(self.child)(x)
                return x, self

        class Child(to.Tree, to.Immutable):
            n: int = to.node()

            def __init__(self):
                self.n = 0

            def __call__(self, x):
                self.n += 1
                return x

        x = np.random.uniform(size=(5, 2))
        module = Parent()

        @to.toplevel_mutable
        def nested_mutable(module: Parent):
            assert module._mutable

            @to.toplevel_mutable
            def nested_fn(module: Parent):
                y, module2 = to.toplevel_mutable(module)(x, False)

                assert module2._mutable

                return module2, module

            module2, module = nested_fn(module)

            assert module._mutable
            assert not module2._mutable

            return module

        module = nested_mutable(module)
        assert not module._mutable

    def test_default(self):
        class A(to.Tree, to.Immutable):
            a: int = to.field(1, node=True)

        tree = A()

        assert tree.a == 1

        tree = jax.tree_map(lambda x: 2 * x, tree)

        assert tree.a == 2

    def test_default_factory(self):
        class A(to.Tree, to.Immutable):
            a: int = to.field(node=True, default_factory=lambda: 1)

        tree = A()

        assert tree.a == 1

        tree = jax.tree_map(lambda x: 2 * x, tree)

        assert tree.a == 2

    def test_flatten(self):

        mlp = MLP(2, 3, 5)

        flat = jax.tree_util.tree_leaves(mlp)

        assert len(flat) == 6

    def test_flatten_slice(self):

        mlp = to.filter(MLP(2, 3, 5), State)

        flat = jax.tree_util.tree_leaves(mlp)

        assert len(flat) == 2

    def test_flatten_slice_merging(self):

        mlp = to.filter(MLP(2, 3, 5), State)

        flat = jax.tree_util.tree_flatten(mlp, lambda x: isinstance(x, to.Nothing))[0]

        assert len(flat) == 6

    def test_is_tree(self):

        mlp = MLP(2, 3, 5)

        @jax.jit
        def idfn(x):
            return x

        assert not isinstance(mlp.linear1.w, jax.Array)
        assert not isinstance(mlp.linear1.b, jax.Array)
        assert not isinstance(mlp.linear1.n, jax.Array)

        assert not isinstance(mlp.linear2.w, jax.Array)
        assert not isinstance(mlp.linear2.b, jax.Array)
        assert not isinstance(mlp.linear1.n, jax.Array)

        mlp = idfn(mlp)

        assert isinstance(mlp.linear1.w, jax.Array)
        assert isinstance(mlp.linear1.b, jax.Array)
        assert isinstance(mlp.linear1.n, jax.Array)

        assert isinstance(mlp.linear2.w, jax.Array)
        assert isinstance(mlp.linear2.b, jax.Array)
        assert isinstance(mlp.linear2.n, jax.Array)

    def test_update_field_metadata(self):

        mlp = MLP(2, 3, 5)

        mlp = mlp.replace(linear1=mlp.linear1.update_field_metadata("w", node=False))

        @jax.jit
        def idfn(x):
            return x

        assert not isinstance(mlp.linear1.w, jax.Array)
        assert not isinstance(mlp.linear1.b, jax.Array)
        assert not isinstance(mlp.linear1.n, jax.Array)

        assert not isinstance(mlp.linear2.w, jax.Array)
        assert not isinstance(mlp.linear2.b, jax.Array)
        assert not isinstance(mlp.linear1.n, jax.Array)

        mlp = idfn(mlp)

        assert not isinstance(mlp.linear1.w, jax.Array)
        assert isinstance(mlp.linear1.b, jax.Array)
        assert isinstance(mlp.linear1.n, jax.Array)

        assert isinstance(mlp.linear2.w, jax.Array)
        assert isinstance(mlp.linear2.b, jax.Array)
        assert isinstance(mlp.linear2.n, jax.Array)

    def test_filter(self):

        mlp = MLP(2, 3, 5)

        # params
        mlp_params = to.filter(mlp, Parameter)

        assert not isinstance(mlp_params.linear1.w, to.Nothing)
        assert not isinstance(mlp_params.linear1.b, to.Nothing)
        assert isinstance(mlp_params.linear1.n, to.Nothing)

        assert not isinstance(mlp_params.linear2.w, to.Nothing)
        assert not isinstance(mlp_params.linear2.b, to.Nothing)
        assert isinstance(mlp_params.linear2.n, to.Nothing)

        # states
        mlp_states = to.filter(mlp, State)

        assert isinstance(mlp_states.linear1.w, to.Nothing)
        assert isinstance(mlp_states.linear1.b, to.Nothing)
        assert not isinstance(mlp_states.linear1.n, to.Nothing)

        assert isinstance(mlp_states.linear2.w, to.Nothing)
        assert isinstance(mlp_states.linear2.b, to.Nothing)
        assert not isinstance(mlp_states.linear2.n, to.Nothing)

    def test_update(self):

        mlp = MLP(2, 3, 5)

        mlp_params = to.filter(mlp, Parameter)
        mlp_states = to.filter(mlp, State)

        mlp_next = to.merge(mlp_params, mlp_states)

        assert not isinstance(mlp_next.linear1.w, to.Nothing)
        assert not isinstance(mlp_next.linear1.b, to.Nothing)
        assert not isinstance(mlp_next.linear1.n, to.Nothing)

        assert not isinstance(mlp_next.linear2.w, to.Nothing)
        assert not isinstance(mlp_next.linear2.b, to.Nothing)
        assert not isinstance(mlp_next.linear2.n, to.Nothing)

    def test_update_not_inplace(self):

        mlp = MLP(2, 3, 5)

        mlp_params = to.filter(mlp, Parameter)
        mlp_states = to.filter(mlp, State)

        to.merge(mlp_params, mlp_states)

        assert not isinstance(mlp_params.linear1.w, to.Nothing)
        assert not isinstance(mlp_params.linear1.b, to.Nothing)
        assert isinstance(mlp_params.linear1.n, to.Nothing)

        assert not isinstance(mlp_params.linear2.w, to.Nothing)
        assert not isinstance(mlp_params.linear2.b, to.Nothing)
        assert isinstance(mlp_params.linear2.n, to.Nothing)

    def test_list(self):
        class LinearList(to.Tree, to.Immutable):
            params: tp.List[np.ndarray] = Parameter.node()

            def __init__(self, din, dout, name="linear"):
                super().__init__()

                self.din = din
                self.dout = dout
                self.params = [
                    np.random.uniform(size=(din, dout)),
                    np.random.uniform(size=(dout,)),
                ]
                self.name = name

        linear = LinearList(2, 3, name="mlp")

        @jax.jit
        def idfn(x):
            return x

        assert not isinstance(linear.params[0], jax.Array)
        assert not isinstance(linear.params[1], jax.Array)

        linear = idfn(linear)

        assert isinstance(linear.params[0], jax.Array)
        assert isinstance(linear.params[1], jax.Array)

    def test_treelist(self):
        class MLP(to.Tree, to.Immutable):
            linears: tp.List[Linear]

            def __init__(self, din, dmid, dout, name="mlp"):
                super().__init__()
                self.linears = [
                    Linear(din, dmid, name="linear1"),
                    Linear(dmid, dout, name="linear2"),
                ]

        mlp = MLP(2, 3, 5)

        @jax.jit
        def idfn(x):
            return x

        assert not isinstance(mlp.linears[0].w, jax.Array)
        assert not isinstance(mlp.linears[0].b, jax.Array)
        assert not isinstance(mlp.linears[0].n, jax.Array)

        assert not isinstance(mlp.linears[1].w, jax.Array)
        assert not isinstance(mlp.linears[1].b, jax.Array)
        assert not isinstance(mlp.linears[1].n, jax.Array)

        mlp = idfn(mlp)

        assert isinstance(mlp.linears[0].w, jax.Array)
        assert isinstance(mlp.linears[0].b, jax.Array)
        assert isinstance(mlp.linears[0].n, jax.Array)

        assert isinstance(mlp.linears[1].w, jax.Array)
        assert isinstance(mlp.linears[1].b, jax.Array)
        assert isinstance(mlp.linears[1].n, jax.Array)

    def test_static_annotation(self):
        class Mod(to.Tree, to.Immutable):
            a: Linear
            b: Linear = to.static()

            def __init__(self):
                super().__init__()
                self.a = Linear(3, 4)
                self.b = Linear(3, 4)

        mod = Mod()

        mod: Mod = jax.tree_map(lambda x: "abc", mod)

        assert len(jax.tree_util.tree_leaves(mod)) == 3

        assert isinstance(mod.a.w, str)
        assert isinstance(mod.a.b, str)
        assert isinstance(mod.a.n, str)

        assert not isinstance(mod.b.w, str)
        assert not isinstance(mod.b.b, str)
        assert not isinstance(mod.b.n, str)

    def test_auto_annotations(self):
        class MLP(to.Tree, to.Immutable):
            def __init__(self, din, dmid, dout, name="mlp"):
                super().__init__()
                self.din = din
                self.dmid = dmid
                self.dout = dout
                self.name = name

                self.linear1 = Linear(din, dmid, name="linear1")
                self.linear2 = Linear(dmid, dout, name="linear2")

        mlp = MLP(2, 3, 5)

        assert "linear1" in mlp.field_metadata

    def test_auto_annotations_inserted(self):
        class MLP(to.Tree, to.Immutable):
            linear3: tp.Optional[Linear] = None

            def __init__(self, din, dmid, dout, name="mlp"):
                super().__init__()
                self.din = din
                self.dmid = dmid
                self.dout = dout
                self.name = name

                self.linear1 = Linear(din, dmid, name="linear1")
                self.linear2 = Linear(dmid, dout, name="linear2")

        mlp = MLP(2, 3, 5)

        mlp = mlp.replace(linear3=Linear(7, 8, name="linear3"))

        jax.tree_util.tree_leaves(mlp)  # force flatten

        assert "linear3" in mlp.field_metadata

    def test_auto_annotations_static(self):
        class MLP(to.Tree, to.Immutable):
            linear2: Linear = to.static()

            def __init__(self, din, dmid, dout, name="mlp"):
                super().__init__()
                self.din = din
                self.dmid = dmid
                self.dout = dout
                self.name = name

                self.linear1 = Linear(din, dmid, name="linear1")
                self.linear2 = Linear(dmid, dout, name="linear2")

        mlp = MLP(2, 3, 5)

        assert "linear1" in mlp.field_metadata
        assert not mlp.field_metadata["linear2"].node

    def test_treex_filter(self):

        tree = dict(a=1, b=Linear(3, 4))

        tree2 = to.filter(tree, Parameter)
        assert isinstance(tree2["a"], to.Nothing)

        tree2 = to.filter(tree, lambda field: isinstance(field.value, int))
        assert tree2["a"] == 1

    def test_module_map(self):
        class A(to.Tree, to.Immutable):
            def __init__(self):
                super().__init__()
                self.a = 1

        module = A()

        def map_fn(x):
            x.a = 2

        module2 = to.apply(map_fn, module)
        assert module.a == 1
        assert module2.a == 2

    def test_hashable(self):
        class A(to.Tree, to.Immutable):
            array: to.Hashable[np.ndarray]

            def __init__(self, array: np.ndarray):
                self.array = to.Hashable(array)

        tree = A(np.array(1))

        n = 0

        @jax.jit
        def f(tree):
            nonlocal n
            n += 1

        f(tree)
        assert n == 1

        f(tree)
        assert n == 1

        tree = tree.replace(array=to.Hashable(np.array(2)))
        f(tree)

        assert n == 2

    def test_generics(self):
        class A(to.Tree, to.Immutable):
            w: jnp.ndarray = Parameter.node()

            def __init__(self, w: jnp.ndarray):
                self.w = w

        class MyTree(to.Tree, to.Immutable):
            tree_or_array: tp.List[tp.Union[jnp.ndarray, A]]

            def __init__(self, tree_or_array: tp.List[tp.Union[jnp.ndarray, A]]):
                self.tree_or_array = tree_or_array

        module = MyTree(
            [
                jnp.ones(shape=[10, 5]),
                A(jnp.array([10.0])),
            ]
        )
        assert isinstance(module.tree_or_array[1], A)

        params = to.filter(module, Parameter)

        assert module.field_metadata["tree_or_array"].kind == type(None)
        assert module.tree_or_array[1].field_metadata["w"].kind == Parameter

        with to.add_field_info():
            infos = jax.tree_util.tree_leaves(module)

        assert infos[0].kind == type(None)
        assert infos[1].kind == Parameter

    def test_update_all_fields(self):
        class A(to.Tree, to.Immutable):
            x: int = to.field(node=True)
            y: int = to.field(node=False)

            def __init__(self, x: int, y: int):
                self.x = x
                self.y = y

        a1 = A(1, 2)
        a2 = A(to.NOTHING, 4)
        a3 = A(5, to.NOTHING)

        aout = to.merge(a1, a2, a3)

        assert aout.x == 5
        assert aout.y == 4

    def test_update_normal(self):
        """
        In this test the static part of `a1` is use of `aout` since `flatten_mode` is set to `normal`.
        """

        class A(to.Tree, to.Immutable):
            x: int = to.field(node=True)
            y: int = to.field(node=False)

            def __init__(self, x: int, y: int):
                self.x = x
                self.y = y

        a1 = A(1, 2)
        a2 = A(to.NOTHING, 2)
        a3 = A(5, 2)

        aout = to.merge(a1, a2, a3, flatten_mode=to.FlattenMode.normal)

        assert aout.x == 5
        assert aout.y == 2

    def test_update_normal_ignore_static(self):
        """
        ignore_static enables you to ignore differences in the static structure of the trees.
        """

        class A(to.Tree, to.Immutable):
            x: int = to.field(node=True)
            y: int = to.field(node=False)

            def __init__(self, x: int, y: int):
                self.x = x
                self.y = y

        a1 = A(1, 2)
        a2 = A(to.NOTHING, 3)
        a3 = A(5, to.NOTHING)

        aout = to.merge(
            a1, a2, a3, flatten_mode=to.FlattenMode.normal, ignore_static=True
        )

        assert aout.x == 5
        assert aout.y == 2

    def test_update_normal_ignore_static_false_raises(self):
        """
        ignore_static enables you to ignore differences in the static structure of the trees.
        """

        class A(to.Tree, to.Immutable):
            x: int = to.field(node=True)
            y: int = to.field(node=False)

            def __init__(self, x: int, y: int):
                self.x = x
                self.y = y

        a1 = A(1, 2)
        a2 = A(to.NOTHING, 3)
        a3 = A(5, to.NOTHING)

        with pytest.raises(ValueError):
            aout = to.merge(
                a1, a2, a3, flatten_mode=to.FlattenMode.normal, ignore_static=False
            )

    def test_jit_on_method(self):
        n = 0

        class A(to.Tree, to.Immutable):
            x: int = to.field(node=True)

            def __init__(self, x: int):
                self.x = x

            @jax.jit
            @to.mutable
            def f(self):
                nonlocal n
                n += 1
                self.x = self.x + 1

        a = A(1)

        _, a = a.f()

        assert a.x == 2
        assert n == 1

        _, a = a.f()
        assert a.x == 3
        assert n == 1

    def test_to_dict(self):
        @dataclass
        class A(to.Tree, to.Immutable):
            x: int = to.field(node=True)
            y: str = to.field(node=False)

        @dataclass
        class B(to.Tree, to.Immutable):
            list_of_a: tp.List[A] = to.field(node=True)
            dict_of_a: tp.Dict[str, A] = to.field(node=False)

        a1 = A(1, "abc")
        a2 = A(3, "def")
        a3 = A(5, "ghi")
        a4 = A(7, "jkl")
        a5 = A(9, "mno")
        a6 = A(11, "pqr")

        b = B([a1, a2, a3], {"a": a4, "b": a5, "c": a6})

        d = to.to_dict(b)

        assert d == {
            "list_of_a": [
                {"x": 1, "y": "abc"},
                {"x": 3, "y": "def"},
                {"x": 5, "y": "ghi"},
            ],
            "dict_of_a": {
                "a": {"x": 7, "y": "jkl"},
                "b": {"x": 9, "y": "mno"},
                "c": {"x": 11, "y": "pqr"},
            },
        }

    def test_to_dict_type_info(self):
        @dataclass
        class A(to.Tree, to.Immutable):
            x: int = to.field(node=True)
            y: str = to.field(node=False)

        @dataclass
        class B(to.Tree, to.Immutable):
            list_of_a: tp.List[A] = to.field(node=True)
            dict_of_a: tp.Dict[str, A] = to.field(node=False)

        a1 = A(1, "abc")
        a2 = A(3, "def")
        a3 = A(5, "ghi")
        a4 = A(7, "jkl")
        a5 = A(9, "mno")
        a6 = A(11, "pqr")

        b = B([a1, a2, a3], {"a": a4, "b": a5, "c": a6})

        d = to.to_dict(b, type_info=True)

        assert d == {
            "list_of_a": [
                {"x": 1, "y": "abc", "__type__": A},
                {"x": 3, "y": "def", "__type__": A},
                {"x": 5, "y": "ghi", "__type__": A},
                list,
            ],
            "dict_of_a": {
                "a": {"x": 7, "y": "jkl", "__type__": A},
                "b": {"x": 9, "y": "mno", "__type__": A},
                "c": {"x": 11, "y": "pqr", "__type__": A},
                "__type__": dict,
            },
            "__type__": B,
        }

    def test_to_dict_field_info(self):
        @dataclass
        class A(to.Tree, to.Immutable):
            x: int = to.field(node=True)
            y: str = to.field(node=False)

        @dataclass
        class B(to.Tree, to.Immutable):
            list_of_a: tp.List[A] = to.field(node=True)
            dict_of_a: tp.Dict[str, A] = to.field(node=False)

        a1 = A(1, "abc")
        a2 = A(3, "def")
        a3 = A(5, "ghi")
        a4 = A(7, "jkl")
        a5 = A(9, "mno")
        a6 = A(11, "pqr")

        b = B([a1, a2, a3], {"a": a4, "b": a5, "c": a6})

        d = to.to_dict(b, field_info=True)

        assert isinstance(d["list_of_a"][0]["x"], to.FieldInfo)
        assert isinstance(d["list_of_a"][0]["y"], to.FieldInfo)
        assert isinstance(d["list_of_a"][1]["x"], to.FieldInfo)
        assert isinstance(d["list_of_a"][1]["y"], to.FieldInfo)
        assert isinstance(d["list_of_a"][2]["x"], to.FieldInfo)
        assert isinstance(d["list_of_a"][2]["y"], to.FieldInfo)

        assert isinstance(d["dict_of_a"]["a"]["x"], to.FieldInfo)
        assert isinstance(d["dict_of_a"]["a"]["y"], to.FieldInfo)
        assert isinstance(d["dict_of_a"]["b"]["x"], to.FieldInfo)
        assert isinstance(d["dict_of_a"]["b"]["y"], to.FieldInfo)
        assert isinstance(d["dict_of_a"]["c"]["x"], to.FieldInfo)
        assert isinstance(d["dict_of_a"]["c"]["y"], to.FieldInfo)

    def test_to_string(self):
        @dataclass
        class A(to.Tree, to.Immutable):
            x: np.ndarray = to.field(node=True, kind=Parameter)
            y: str = to.field(node=False, kind=State)

        @dataclass
        class B(to.Tree, to.Immutable):
            list_of_a: tp.List[A] = to.field(node=True)
            dict_of_a: tp.Dict[str, A] = to.field(node=False)

        a1 = A(np.zeros((2, 4)), "abc")
        a2 = A(jnp.zeros((2, 4)), "def")
        a3 = A(np.zeros((2, 4)), "ghi")
        a4 = A(jnp.zeros((2, 4)), "jkl")
        a5 = A(np.zeros((2, 4)), "mno")
        a6 = A(jnp.zeros((2, 4)), "pqr")

        b = B([a1, a2, a3], {"a": a4, "b": a5, "c": a6})

        rep = to.to_string(b, color=True)

        print(rep)

    def test_compact(self):
        class Linear(to.Tree, to.Compact, to.Immutable):
            w: tp.Optional[np.ndarray] = Parameter.node(None)
            b: tp.Optional[np.ndarray] = Parameter.node(None)
            n: tp.Optional[int] = State.node(None)

            def __init__(self, din, dout, name="linear"):
                self.din = din
                self.dout = dout
                self.name = name

            @to.compact
            def __call__(self):
                if self.first_run:
                    self.w = np.random.uniform(size=(self.din, self.dout))
                    self.b = np.random.uniform(size=(self.dout,))
                    self.n = 1

        class MLP(to.Tree, to.Immutable):
            def __init__(self, din, dmid, dout, name="mlp"):
                self.din = din
                self.dmid = dmid
                self.dout = dout
                self.name = name

            @to.compact
            def __call__(self):
                Linear(self.din, self.dmid, name="linear1")()
                Linear(self.dmid, self.dout, name="linear2")()

        mlp: MLP = MLP(2, 4, 3)
        mlp = to.mutable(mlp)()[1]
        mlp = to.mutable(mlp)()[1]

        assert mlp.linear1.w.shape == (2, 4)
        assert mlp.linear1.b.shape == (4,)
        assert mlp.linear1.n == 1

        assert mlp.linear2.w.shape == (4, 3)
        assert mlp.linear2.b.shape == (3,)
        assert mlp.linear2.n == 1

        assert mlp.linear1.name == "linear1"
        assert mlp.linear2.name == "linear2"
        assert mlp.name == "mlp"

        flat = jax.tree_util.tree_leaves(mlp)

        assert len(flat) == 6

        # params
        mlp_params = to.filter(mlp, Parameter)

        assert not isinstance(mlp_params.linear1.w, to.Nothing)
        assert not isinstance(mlp_params.linear1.b, to.Nothing)
        assert isinstance(mlp_params.linear1.n, to.Nothing)

        assert not isinstance(mlp_params.linear2.w, to.Nothing)
        assert not isinstance(mlp_params.linear2.b, to.Nothing)
        assert isinstance(mlp_params.linear2.n, to.Nothing)

        # states
        mlp_states = to.filter(mlp, State)

        assert isinstance(mlp_states.linear1.w, to.Nothing)
        assert isinstance(mlp_states.linear1.b, to.Nothing)
        assert not isinstance(mlp_states.linear1.n, to.Nothing)

        assert isinstance(mlp_states.linear2.w, to.Nothing)
        assert isinstance(mlp_states.linear2.b, to.Nothing)
        assert not isinstance(mlp_states.linear2.n, to.Nothing)

    def test_compact_sugar(self):
        class Linear(to.Tree, to.Compact, to.Immutable):
            w: tp.Optional[np.ndarray] = Parameter.node(None)
            b: tp.Optional[np.ndarray] = Parameter.node(None)
            n: tp.Optional[int] = State.node(None)

            def __init__(self, din, dout, name="linear"):
                self.din = din
                self.dout = dout
                self.name = name

            @to.compact
            def __call__(self):
                if self.first_run:
                    self.w = np.random.uniform(size=(self.din, self.dout))
                    self.b = np.random.uniform(size=(self.dout,))
                    self.n = 1

        class MLP(to.Tree, to.Immutable):
            def __init__(self, din, dmid, dout, name="mlp"):
                self.din = din
                self.dmid = dmid
                self.dout = dout
                self.name = name

            @to.compact
            def __call__(self):
                Linear(self.din, self.dmid, name="linear1")()
                Linear(self.dmid, self.dout, name="linear2")()

        mlp: MLP = MLP(2, 4, 3)
        mlp = to.mutable(mlp)()[1]
        mlp = to.mutable(mlp)()[1]

        assert mlp.linear1.w.shape == (2, 4)
        assert mlp.linear1.b.shape == (4,)
        assert mlp.linear1.n == 1

        assert mlp.linear2.w.shape == (4, 3)
        assert mlp.linear2.b.shape == (3,)
        assert mlp.linear2.n == 1

        assert mlp.linear1.name == "linear1"
        assert mlp.linear2.name == "linear2"
        assert mlp.name == "mlp"

        flat = jax.tree_util.tree_leaves(mlp)

        assert len(flat) == 6

        # params
        mlp_params = to.filter(mlp, Parameter)

        assert not isinstance(mlp_params.linear1.w, to.Nothing)
        assert not isinstance(mlp_params.linear1.b, to.Nothing)
        assert isinstance(mlp_params.linear1.n, to.Nothing)

        assert not isinstance(mlp_params.linear2.w, to.Nothing)
        assert not isinstance(mlp_params.linear2.b, to.Nothing)
        assert isinstance(mlp_params.linear2.n, to.Nothing)

        # states
        mlp_states = to.filter(mlp, State)

        assert isinstance(mlp_states.linear1.w, to.Nothing)
        assert isinstance(mlp_states.linear1.b, to.Nothing)
        assert not isinstance(mlp_states.linear1.n, to.Nothing)

        assert isinstance(mlp_states.linear2.w, to.Nothing)
        assert isinstance(mlp_states.linear2.b, to.Nothing)
        assert not isinstance(mlp_states.linear2.n, to.Nothing)

    def test_compact_naming(self):
        class Linear(to.Tree, to.Compact, to.Immutable):
            w: tp.Optional[np.ndarray] = Parameter.node(None)
            b: tp.Optional[np.ndarray] = Parameter.node(None)
            n: tp.Optional[int] = State.node(None)

            def __init__(self, din, dout, name="linear"):
                self.din = din
                self.dout = dout
                self.name = name

            @to.compact
            def __call__(self):
                if self.first_run:
                    self.w = np.random.uniform(size=(self.din, self.dout))
                    self.b = np.random.uniform(size=(self.dout,))
                    self.n = 1

        class MLP(to.Tree, to.Immutable):
            def __init__(self, din, dmid, dout, name="mlp"):
                self.din = din
                self.dmid = dmid
                self.dout = dout
                self.name = name

            @to.compact
            def __call__(self):
                Linear(self.din, self.dmid)()
                Linear(self.dmid, self.dout)()

        mlp: MLP = MLP(2, 4, 3)
        mlp = to.mutable(mlp)()[1]
        mlp = to.mutable(mlp)()[1]

        assert mlp.linear.w.shape == (2, 4)
        assert mlp.linear.b.shape == (4,)
        assert mlp.linear.n == 1

        assert mlp.linear2.w.shape == (4, 3)
        assert mlp.linear2.b.shape == (3,)
        assert mlp.linear2.n == 1

        assert hasattr(mlp, "linear")
        assert hasattr(mlp, "linear2")

        assert mlp.linear.din == 2
        assert mlp.linear.dout == 4

        assert mlp.linear2.din == 4
        assert mlp.linear2.dout == 3

    def test_compact_new_field(self):
        class Linear(to.Tree, to.Compact, to.Immutable):
            def __init__(self, din, dout, name="linear"):
                self.din = din
                self.dout = dout
                self.name = name

            @to.compact
            def __call__(self):
                self.w = np.random.uniform(size=(self.din, self.dout))

        tree: Linear = Linear(2, 4)

        _, tree = to.mutable(tree)()

        assert not tree.field_metadata["w"].node

    def test_compact_override_ok(self):
        class Linear(to.Tree, to.Compact, to.Immutable):
            w: tp.Optional[np.ndarray] = Parameter.node(None)

            def __init__(self, din, dout, name="linear"):
                self.din = din
                self.dout = dout
                self.name = name

            @to.compact
            def __call__(self):
                if self.first_run:
                    self.w = np.random.uniform(size=(self.din, self.dout))

        class NewLinear(Linear):
            b: tp.Optional[np.ndarray] = Parameter.node(None)

            @to.compact
            def __call__(self):
                if self.first_run:
                    self.b = np.random.uniform(size=(self.dout,))
                super().__call__()

        tree = NewLinear(2, 4)

        tree = to.mutable(tree)()[1]

        assert tree.w.shape == (2, 4)
        assert tree.b.shape == (4,)

    def test_compact_recursion_error(self):
        class Linear(to.Tree, to.Compact, to.Immutable):
            w: tp.Optional[np.ndarray] = Parameter.node(None)

            def __init__(self, din, dout, name="linear"):
                self.din = din
                self.dout = dout
                self.name = name

            @to.compact
            def __call__(self):
                if self.first_run:
                    self.w = np.random.uniform(size=(self.din, self.dout))
                self()

        tree = Linear(2, 4)

        with pytest.raises(RuntimeError):
            tree = to.mutable(tree)()[1]

    def test_not_compact_in_init(self):
        a_init_ran = False
        b_init_ran = False

        class A(to.Tree, to.Immutable):
            def __init__(self) -> None:
                nonlocal a_init_ran
                a_init_ran = True
                assert not to.in_compact()

        class B(to.Tree, to.Immutable):
            def __init__(self) -> None:
                nonlocal b_init_ran
                b_init_ran = True
                assert not to.in_compact()

            @to.compact
            def __call__(self):
                A()

        b = B()

        assert b_init_ran

        b = to.mutable(b)()[1]

        assert a_init_ran
        assert "a" in vars(b)
        assert isinstance(b.a, A)

    def test_with_field_info(self):
        class Parameter:
            @staticmethod
            def fn(x):
                return np.asarray(x**2)

        @dataclass
        class A(to.Tree, to.Immutable):
            x: np.ndarray = to.field(node=True, kind=Parameter)
            y: np.ndarray = to.field(node=True)

        m = A(np.array(2.0), np.array(2.0))

        def field_map_fn(f):
            return f.kind.fn(f.value)

        m1 = to.map(field_map_fn, m, Parameter, field_info=True)

        assert m.x == 2.0
        assert m.y == 2.0
        assert m1.x == 4.0
        assert m1.y == 2.0
        assert type(m) == type(m1)
        assert type(m1.x) == type(m.x)
        assert type(m1.y) == type(m.y)

    def test_uninitialized(self):
        class A(to.Tree, to.Immutable):
            x: int = to.field(node=True)

            def __init__(self, x: tp.Optional[int] = None) -> None:
                if x is not None:
                    self.x = x

        with pytest.raises(TypeError):
            a = A()

        a = A(1)
        assert a.x == 1
