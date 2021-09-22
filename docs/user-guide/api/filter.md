<!-- #### Filter -->
The `filter` function allows you to select a subtree by filtering based on a `kind`, all leaves whose field kind is a subclass of such type are kept, the rest are set to a special `Nothing` value.

```python
tree = MyTree(a=jnp.array(1), b=jnp.array(2))

to.filter(tree, Parameter) # MyTree(a=array(1), b=Nothing)
to.filter(tree, BatchStat) # MyTree(a=Nothing, b=array(2))
```
Since `Nothing` is an empty Pytree it gets ignored by tree operations, this effectively allows you to easily operate on a subset of the fields:

```python
jax.tree_map(lambda x: -x, to.filter(tree, Parameter)) # MyTree(a=array(-1), b=Nothing)
jax.tree_map(lambda x: -x, to.filter(tree, BatchStat)) # MyTree(a=Nothing, b=array([-2]))
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
## multiple filters
The previous could be abbreviated using multiple filters which you can pass as `*args`. For a field to be kept it will required that **all filters pass**. Since passing types by themselves are "kind filters", the previous example could be written as:
```python
# all Parameters whose field name is "kernel"
to.filter(
    tree,
    Parameter,
    lambda field: field.name == "kernel"
) 
# MyTree(a=Nothing, b=Nothing)
```