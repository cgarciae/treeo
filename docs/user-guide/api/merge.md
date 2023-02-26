# Merge

Creates a new Tree with the same structure but its values merged based on the values from the incoming Trees. 

```python
@dataclass
class MyTree(to.Tree):
    a: int = to.field(node=True, kind=Parameter)
    b: int = to.field(node=True, kind=BatchStat)

t1 = MyTree(x=Nothing, y=2, z=3)
t2 = MyTree(x=1, y=Nothing, z=4)

merge(t1, t2) # MyTree(x=1, y=2, z=4)
```

Updates are performed using the following rules:

* For a list of equivalent leaves `l1, l2, ..., ln`, it returns the first non-`Nothing` leaf from right to left.
* If no `flatten_mode()` context manager is active and `flatten_mode` is not given, all fields will be merged.
* If `flatten_mode="normal"` is set then static fields won't be merged and the output will have the exact same static components as the first input (`obj`).

When using `merge` with multiple Trees the following equivalence holds:

```
merge(m1, m2, m3) = merge(m1, merge(m2, m3))
```

If you want to merge the current module instead of creating a new one use `inplace=True`. This is useful when applying transformation inside a method where reassigning `self` is not possible:

```python
def double_params(self):
    # this is not doing what you expect
    self = jax.tree_map(lambda x: 2 * x, self)
```
Instead do this:

```python
def double_params(self):
    doubled = jax.tree_map(lambda x: 2 * x, self)
    merge(self, doubled, inplace=True)
```

If `inplace` is `True`, the input `obj` is mutated and returned. You can only merge inplace if the input `obj` has a `__dict__` attribute, else a `TypeError` is raised.

If `ignore_static` is `True`, static fields (according to the flattening mode) will be bypassed during the merge process, the final output will have the same static components as the first input (`obj`). This strategy is a bit less safe in general as it will flatten all trees using `jax.tree_util.tree_leaves` instead of `PyTreeDef.flatten_up_to`, this skips some checks so it effectively ignores their static components, the only requirement is that the flattened struture of all trees matches.
