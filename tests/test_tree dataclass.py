import typing as tp
from dataclasses import dataclass
from inspect import Parameter

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
import pytest

import treeo as to


class Parameter(to.KindMixin):
    pass


class State(to.KindMixin):
    pass


@dataclass
class Linear(to.Tree):
    din: int
    dout: int
    w: np.ndarray = Parameter.node(init=False)
    b: np.ndarray = Parameter.node(init=False)
    name: str = "linear"
    n: int = State.node(1)

    def __post_init__(self):
        self.w = np.random.uniform(size=(self.din, self.dout))
        self.b = np.random.uniform(size=(self.dout,))


@dataclass
class MLP(to.Tree):
    din: int
    dmid: int
    dout: int
    linear1: Linear = to.node(init=False)
    linear2: Linear = to.node(init=False)
    name: str = "mlp"

    def __post_init__(self):
        self.linear1 = Linear(self.din, self.dmid, name="linear1")
        self.linear2 = Linear(self.dmid, self.dout, name="linear2")


class TestTreeoDataclass:
    def test_default(self):
        @dataclass
        class A(to.Tree):
            a: int = to.field(1, node=True)

        tree = A()

        assert tree.a == 1

        tree = jax.tree_map(lambda x: 2 * x, tree)

        assert tree.a == 2

    def test_default_factory(self):
        @dataclass
        class A(to.Tree):
            a: int = to.field(node=True, default_factory=lambda: 1)

        tree = A()

        assert tree.a == 1

        tree = jax.tree_map(lambda x: 2 * x, tree)

        assert tree.a == 2

    def test_flatten_nothing(self):
        x = [(1, 2), (3, to.Nothing())]
        assert jax.tree_leaves(x) == [1, 2, 3]

        flat_with_nothing = jax.tree_flatten(x, lambda x: isinstance(x, to.Nothing))[0]

        assert flat_with_nothing == [1, 2, 3, to.Nothing()]

    def test_flatten(self):

        mlp = MLP(2, 3, 5)
        mlp = MLP(2, 3, 5)

        flat = jax.tree_leaves(mlp)

        assert len(flat) == 6

    def test_flatten_slice(self):

        mlp = to.filter(MLP(2, 3, 5), State)

        flat = jax.tree_leaves(mlp)

        assert len(flat) == 2

    def test_flatten_slice_merging(self):

        mlp = to.filter(MLP(2, 3, 5), State)

        flat = jax.tree_flatten(mlp, lambda x: isinstance(x, to.Nothing))[0]

        assert len(flat) == 6

    def test_is_tree(self):

        mlp = MLP(2, 3, 5)

        @jax.jit
        def idfn(x):
            return x

        assert not isinstance(mlp.linear1.w, jnp.DeviceArray)
        assert not isinstance(mlp.linear1.b, jnp.DeviceArray)
        assert not isinstance(mlp.linear1.n, jnp.DeviceArray)

        assert not isinstance(mlp.linear2.w, jnp.DeviceArray)
        assert not isinstance(mlp.linear2.b, jnp.DeviceArray)
        assert not isinstance(mlp.linear1.n, jnp.DeviceArray)

        mlp = idfn(mlp)

        assert isinstance(mlp.linear1.w, jnp.DeviceArray)
        assert isinstance(mlp.linear1.b, jnp.DeviceArray)
        assert isinstance(mlp.linear1.n, jnp.DeviceArray)

        assert isinstance(mlp.linear2.w, jnp.DeviceArray)
        assert isinstance(mlp.linear2.b, jnp.DeviceArray)
        assert isinstance(mlp.linear2.n, jnp.DeviceArray)

    def test_update_field_metadata(self):

        mlp = MLP(2, 3, 5)

        mlp.linear1 = mlp.linear1.update_field_metadata("w", node=False)

        @jax.jit
        def idfn(x):
            return x

        assert not isinstance(mlp.linear1.w, jnp.DeviceArray)
        assert not isinstance(mlp.linear1.b, jnp.DeviceArray)
        assert not isinstance(mlp.linear1.n, jnp.DeviceArray)

        assert not isinstance(mlp.linear2.w, jnp.DeviceArray)
        assert not isinstance(mlp.linear2.b, jnp.DeviceArray)
        assert not isinstance(mlp.linear1.n, jnp.DeviceArray)

        mlp = idfn(mlp)

        assert not isinstance(mlp.linear1.w, jnp.DeviceArray)
        assert isinstance(mlp.linear1.b, jnp.DeviceArray)
        assert isinstance(mlp.linear1.n, jnp.DeviceArray)

        assert isinstance(mlp.linear2.w, jnp.DeviceArray)
        assert isinstance(mlp.linear2.b, jnp.DeviceArray)
        assert isinstance(mlp.linear2.n, jnp.DeviceArray)

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
        class LinearList(to.Tree):
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

        assert not isinstance(linear.params[0], jnp.DeviceArray)
        assert not isinstance(linear.params[1], jnp.DeviceArray)

        linear = idfn(linear)

        assert isinstance(linear.params[0], jnp.DeviceArray)
        assert isinstance(linear.params[1], jnp.DeviceArray)

    def test_treelist(self):
        class MLP(to.Tree):
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

        assert not isinstance(mlp.linears[0].w, jnp.DeviceArray)
        assert not isinstance(mlp.linears[0].b, jnp.DeviceArray)
        assert not isinstance(mlp.linears[0].n, jnp.DeviceArray)

        assert not isinstance(mlp.linears[1].w, jnp.DeviceArray)
        assert not isinstance(mlp.linears[1].b, jnp.DeviceArray)
        assert not isinstance(mlp.linears[1].n, jnp.DeviceArray)

        mlp = idfn(mlp)

        assert isinstance(mlp.linears[0].w, jnp.DeviceArray)
        assert isinstance(mlp.linears[0].b, jnp.DeviceArray)
        assert isinstance(mlp.linears[0].n, jnp.DeviceArray)

        assert isinstance(mlp.linears[1].w, jnp.DeviceArray)
        assert isinstance(mlp.linears[1].b, jnp.DeviceArray)
        assert isinstance(mlp.linears[1].n, jnp.DeviceArray)

    def test_static_annotation(self):
        class Mod(to.Tree):
            a: Linear
            b: Linear = to.static()

            def __init__(self):
                super().__init__()
                self.a = Linear(3, 4)
                self.b = Linear(3, 4)

        mod = Mod()

        mod: Mod = jax.tree_map(lambda x: "abc", mod)

        assert len(jax.tree_leaves(mod)) == 3

        assert isinstance(mod.a.w, str)
        assert isinstance(mod.a.b, str)
        assert isinstance(mod.a.n, str)

        assert not isinstance(mod.b.w, str)
        assert not isinstance(mod.b.b, str)
        assert not isinstance(mod.b.n, str)

    def test_auto_annotations(self):
        class MLP(to.Tree):
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
        class MLP(to.Tree):
            def __init__(self, din, dmid, dout, name="mlp"):
                super().__init__()
                self.din = din
                self.dmid = dmid
                self.dout = dout
                self.name = name

                self.linear1 = Linear(din, dmid, name="linear1")
                self.linear2 = Linear(dmid, dout, name="linear2")

        mlp = MLP(2, 3, 5)

        mlp.linear3 = Linear(7, 8, name="linear3")

        jax.tree_leaves(mlp)  # force flatten

        assert "linear3" in mlp.field_metadata

    def test_auto_annotations_static(self):
        class MLP(to.Tree):
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

    def test_annotations_missing_field_no_error(self):
        class MLP(to.Tree):
            linear3: Linear  # missing field

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
        assert "linear2" in mlp.field_metadata

    def test_treex_filter(self):

        tree = dict(a=1, b=Linear(3, 4))

        tree2 = to.filter(tree, Parameter)
        assert isinstance(tree2["a"], to.Nothing)

        tree2 = to.filter(tree, lambda field: isinstance(field.value, int))
        assert tree2["a"] == 1

    def test_module_map(self):
        class A(to.Tree):
            def __init__(self):
                super().__init__()
                self.a = 1

        module = A()

        def map_fn(x):
            x.a = 2

        module2 = to.apply(map_fn, module)

        assert module.a == 1
        assert module2.a == 2

    def test_opaque(self):
        @dataclass
        class A(to.Tree):
            id: to.Opaque[str]

            @classmethod
            def new(cls, id: str):
                return cls(id=to.Opaque(id, None))

        a1 = A.new("1")
        a2 = A.new("2")

        n = 0

        @jax.jit
        def f(a):
            nonlocal n
            n += 1

        f(a1)
        f(a2)

        assert n == 1

    def test_opaque_field(self):
        @dataclass
        class A(to.Tree):
            id: str = to.field(node=False, opaque=True)

        a1 = A("1")
        a2 = A("2")

        n = 0

        @jax.jit
        def f(a):
            nonlocal n
            n += 1

        f(a1)
        f(a2)

        assert n == 1

    def test_hashable(self):
        @dataclass
        class A(to.Tree):
            array: to.Hashable[np.ndarray]

            @classmethod
            def new(cls, array: np.ndarray):
                return cls(to.Hashable(array))

        tree = A.new(np.array(1))

        n = 0

        @jax.jit
        def f(tree):
            nonlocal n
            n += 1

        f(tree)
        assert n == 1

        f(tree)
        assert n == 1

        tree.array = to.Hashable(np.array(2))
        f(tree)

        assert n == 2

    def test_generics(self):
        @dataclass
        class A(to.Tree):
            w: jnp.ndarray = Parameter.node()

        @dataclass
        class MyTree(to.Tree):
            tree_or_array: tp.List[tp.Union[jnp.ndarray, A]]

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
            infos = jax.tree_leaves(module)

        assert infos[0].kind == type(None)
        assert infos[1].kind == Parameter

    def test_static(self):
        # @dataclass
        class A(to.Tree):
            _initialized: bool = to.static(False)

        @dataclass
        class B(A):
            pass

        tree = B()

        assert tree._initialized is False
