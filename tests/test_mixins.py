import typing as tp
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import treeo as to


class TestMixins:
    def test_apply(self):
        @dataclass
        class SomeTree(to.Tree, to.Apply):
            x: int = to.node()

        tree = SomeTree(x=1)

        def f(tree: SomeTree):
            tree.x = 2

        tree2 = tree.apply(f)

        assert tree.x == 1
        assert tree2.x == 2
