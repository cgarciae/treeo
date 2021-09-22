
<!-- #### Update -->
The `update` function allows you to merge the values of one or more incoming pytrees with the current pytree, this is useful for integrating filtered Trees back into the main Tree.

```python hl_lines="4"
import treeo as to

tree = MyTree(a=jnp.array(1), b=jnp.array(2))
params = to.filter(tree, Parameter) # MyTree(a=array(1), b=Nothing)
negative = jax.tree_map(lambda x: -x, params) # MyTree(a=array(-1), b=Nothing)
tree = to.update(tree, negative) # MyTree(a=array(-1), b=array(2))
```