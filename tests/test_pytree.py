from inspect import Parameter
import typing as tp

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
import pytest

import pytreex as ptx


class Parameter(ptx.FieldMixin):
    pass


class State(ptx.FieldMixin):
    pass


class Linear(ptx.Tree):
    w: np.ndarray = Parameter.node()
    b: np.ndarray = Parameter.node()
    n: int = State.node()

    def __init__(self, din, dout, name="linear"):
        super().__init__()
        self.din = din
        self.dout = dout
        self.w = np.random.uniform(size=(din, dout))
        self.b = np.random.uniform(size=(dout,))
        self.n = 1
        self.name = name


class MLP(ptx.Tree):
    linear1: Linear
    linear2: Linear

    def __init__(self, din, dmid, dout, name="mlp"):
        super().__init__()
        self.din = din
        self.dmid = dmid
        self.dout = dout
        self.name = name

        self.linear1 = Linear(din, dmid, name="linear1")
        self.linear2 = Linear(dmid, dout, name="linear2")


class TestTreex:
    def test_flatten_nothing(self):
        x = [(1, 2), (3, ptx.Nothing())]
        assert jax.tree_leaves(x) == [1, 2, 3]

        flat_with_nothing = jax.tree_flatten(x, lambda x: isinstance(x, ptx.Nothing))[0]

        assert flat_with_nothing == [1, 2, 3, ptx.Nothing()]

    def test_flatten(self):

        mlp = MLP(2, 3, 5)

        flat = jax.tree_leaves(mlp)

        assert len(flat) == 6

    def test_flatten_slice(self):

        mlp = ptx.filter(MLP(2, 3, 5), State)

        flat = jax.tree_leaves(mlp)

        assert len(flat) == 2

    def test_flatten_slice_merging(self):

        mlp = ptx.filter(MLP(2, 3, 5), State)

        flat = jax.tree_flatten(mlp, lambda x: isinstance(x, ptx.Nothing))[0]

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
        mlp_params = ptx.filter(mlp, Parameter)

        assert not isinstance(mlp_params.linear1.w, ptx.Nothing)
        assert not isinstance(mlp_params.linear1.b, ptx.Nothing)
        assert isinstance(mlp_params.linear1.n, ptx.Nothing)

        assert not isinstance(mlp_params.linear2.w, ptx.Nothing)
        assert not isinstance(mlp_params.linear2.b, ptx.Nothing)
        assert isinstance(mlp_params.linear2.n, ptx.Nothing)

        # states
        mlp_states = ptx.filter(mlp, State)

        assert isinstance(mlp_states.linear1.w, ptx.Nothing)
        assert isinstance(mlp_states.linear1.b, ptx.Nothing)
        assert not isinstance(mlp_states.linear1.n, ptx.Nothing)

        assert isinstance(mlp_states.linear2.w, ptx.Nothing)
        assert isinstance(mlp_states.linear2.b, ptx.Nothing)
        assert not isinstance(mlp_states.linear2.n, ptx.Nothing)

    def test_update(self):

        mlp = MLP(2, 3, 5)

        mlp_params = ptx.filter(mlp, Parameter)
        mlp_states = ptx.filter(mlp, State)

        mlp_next = ptx.update(mlp_params, mlp_states)

        assert not isinstance(mlp_next.linear1.w, ptx.Nothing)
        assert not isinstance(mlp_next.linear1.b, ptx.Nothing)
        assert not isinstance(mlp_next.linear1.n, ptx.Nothing)

        assert not isinstance(mlp_next.linear2.w, ptx.Nothing)
        assert not isinstance(mlp_next.linear2.b, ptx.Nothing)
        assert not isinstance(mlp_next.linear2.n, ptx.Nothing)

    def test_update_not_inplace(self):

        mlp = MLP(2, 3, 5)

        mlp_params = ptx.filter(mlp, Parameter)
        mlp_states = ptx.filter(mlp, State)

        ptx.update(mlp_params, mlp_states)

        assert not isinstance(mlp_params.linear1.w, ptx.Nothing)
        assert not isinstance(mlp_params.linear1.b, ptx.Nothing)
        assert isinstance(mlp_params.linear1.n, ptx.Nothing)

        assert not isinstance(mlp_params.linear2.w, ptx.Nothing)
        assert not isinstance(mlp_params.linear2.b, ptx.Nothing)
        assert isinstance(mlp_params.linear2.n, ptx.Nothing)

    def test_list(self):
        class LinearList(ptx.Tree):
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
        class MLP(ptx.Tree):
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
        class Mod(ptx.Tree):
            a: Linear
            b: Linear = ptx.static()

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
        class MLP(ptx.Tree):
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
        class MLP(ptx.Tree):
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
        class MLP(ptx.Tree):
            linear2: Linear = ptx.static()

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
        class MLP(ptx.Tree):
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

        tree2 = ptx.filter(tree, Parameter)
        assert isinstance(tree2["a"], ptx.Nothing)

        tree2 = ptx.filter(tree, lambda field: isinstance(field.value, int))
        assert tree2["a"] == 1

    def test_module_map(self):
        class A(ptx.Tree):
            def __init__(self):
                super().__init__()
                self.a = 1

        module = A()

        def map_fn(x):
            x.a = 2

        module2 = ptx.object_apply(map_fn, module)

        assert module.a == 1
        assert module2.a == 2
