
<!-- ### Non-hashable static fields -->
Static fields are required to be hashable by JAX, so what happens if you want to have a static field that contains a non-hashable value like a numpy or jax array? For example:

```python
import treeo as to

@dataclass
class MyTree(to.Tree):
    table: np.ndarray
```

One solution is to make that field `opaque`. This will work as long as you don't need `jit` and friends to react to changes to its value content for recompiling.  If you do, you can instead use the `to.Hashable(value)` class to wrap around it like this:

```python
@dataclass
class MyTree(to.Tree):
    table: to.Hashable[np.ndarray]
```

The hash from `Hashable` will only depend on object identity but not on the actual `value`, therefore you should treat it as immutable, if you want to update its content you should create a new `Hashable` instance:

```python hl_lines="13"
table = to.Hasable(np.ones((10, 10)))
tree = MyTree(table)

@jax.jit
def do_something(tree: MyTree):
    table_value = tree.table.value # use Hashable.value
    ...

tree = do_something(tree) # compiles
tree = do_something(tree) # uses cache

# update table
module.table = to.Hashable(np.zeros((10, 10)))

tree = do_something(tree) # recompiles
```
**Warning**: If you somehow mutate `Hashable.value` directly, JAX won't know about this and `jit` won't recompile.
