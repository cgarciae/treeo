# Map

Applies a function to all leaves in a pytree using `jax.tree_map`. If `filters` are given then the function will be applied only to the subset of leaves that match the filters. Further, property of the field's kind type can be used within `map` by setting the `field_info` argument as True.

For example, if we want to zero all batch stats we can do:

Example:

```python
@dataclass
class MyTree(to.Tree):
    a: int = to.field(node=True, kind=Parameter)
    b: int = to.field(node=True, kind=BatchStat)

tree = MyTree(a=1, b=2)

to.map(lambda _: 0, tree, BatchStat) # MyTree(a=1, b=0)
```

We could also apply a custom function to those fields with the Parameter type

```python
def parameter_fn(field):
    return f + 1

to.map(parameter_fn, tree, Parameter, field_info = True)  # MyTree(a=2, b=2)

`map` is equivalent to `filter -> tree_map -> merge` in sequence.

If `inplace` is `True`, the input `obj` is mutated and returned. You can only update inplace if the input `obj` has a `__dict__` attribute, else a `TypeError` is raised.
```
