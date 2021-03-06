# Filter
The `filter` function allows you to select a subtree by filtering based on a `kind`, all leaves whose field kind is a subclass of such type are kept, the rest are set to a special `Nothing` value.

```python
@dataclass
class MyTree(to.Tree):
    a: int = to.field(node=True, kind=Parameter)
    b: int = to.field(node=True, kind=BatchStat)

tree = MyTree(a=1, b=2)

to.filter(tree, Parameter) # MyTree(a=1, b=Nothing)
to.filter(tree, BatchStat) # MyTree(a=Nothing, b=2)
```
Since `Nothing` is an empty Pytree it gets ignored by tree operations, this effectively allows you to easily operate on a subset of the fields:

```python
jax.tree_map(lambda x: -x, to.filter(tree, Parameter)) # MyTree(a=-1, b=Nothing)
jax.tree_map(lambda x: -x, to.filter(tree, BatchStat)) # MyTree(a=Nothing, b=-2)
```

## filter predicates
If you need to do more complex filtering, you can pass callables with the signature 

```
FieldInfo -> bool
``` 

instead of types:

```python
# all Parameters whose field name is "kernel"
to.filter(
    tree,
    lambda field: issubclass(field.kind, Parameter) 
    and field.name == "kernel"
) 
# MyTree(a=Nothing, b=Nothing)

```

Since `filter` works with pytrees in general, the following is possible:

```python
def array_like(field):
    return hasattr(field.value, "shape") and hasattr(field.value, "dtype")

tree = [1, np.2, jnp.array([3.0, 4.0])]

to.filter(tree, array_like) # [Nothing, np.2, jnp.array([3.0, 4.0])]
```

## multiple filters
You can some queries by using multiple filters as `*args`. For a field to be kept it will required that **all filters pass**. Since passing types by themselves are "kind filters", one of the previous examples could be written as:
```python
# all Parameters whose field name is "kernel"
to.filter(
    tree,
    Parameter,
    lambda field: field.name == "kernel"
) 
# MyTree(a=Nothing, b=Nothing)
```

## inplace
If `inplace` is `True`, the input `obj` is mutated and returned. You can only update inplace if the input `obj` has a `__dict__` attribute, else a `TypeError` is raised.