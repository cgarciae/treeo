
<!-- #### Map -->
The `map` function provides a convenient way to map a function over the fields of a pytree:

```python hl_lines="5"
import treeo as to

tree = MyTree(a=jnp.array(1), b=jnp.array(2))
params = to.filter(tree, Parameter) # MyTree(a=array(1), b=Nothing)
negative = to.map(lambda x: -x, params) # MyTree(a=array(-1), b=Nothing)
tree = to.update(tree, negative) # MyTree(a=array(-1), b=array(2))
```

Up to this point `map` is behaving just like `jax.tree_map`, however, the pattern in the previous example is so common that `map`s main use is providing a shortcut for applying `filter -> tree_map -> update` in sequence:

```python hl_lines="2"
tree = MyTree(a=jnp.array(1), b=jnp.array(2))
tree = to.map(lambda x: -x, tree, Parameter) # MyTree(a=array(-1), b=array(2))
```

As shown here, `map` accepts the same `*args` as `filter` and calls `update` at the end if filters are given.
