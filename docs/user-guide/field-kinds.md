<!-- ### Field kinds -->
Kinds are associated types that give semantic meaning to a field (what it represents). A kind is just a type you pass to `field` via its `kind` argument. Kinds are mostly useful as metadata filtering via [treeo.filter](#filter). For example, here is a possible definition for a `BatchNorm` module using kinds:

```python
import treeo as to

class Parameter: pass
class BatchStat: pass

class BatchNorm(to.Tree):
    scale: jnp.ndarray = to.field(node=True, kind=Parameter)
    bias: jnp.ndarray = to.field(node=True, kind=Parameter)

    mean: jnp.ndarray = to.field(node=True, kind=BatchStat)
    var: jnp.ndarray = to.field(node=True, kind=BatchStat)
    ...
```
Now with this definition you use `filter` to select specific kind of fields:

```python
model = BatchNorm(...) 

# BatchNorm(scale=array(...), bias=array(...), mean=Nothing, var=Nothing)
params = to.filter(model, Parameter) # filter by kind

# BatchNorm(scale=Nothing, bias=Nothing, mean=array(...), var=array(...))
batch_stats = to.filter(model, BatchStat)
```

This can be very useful to operate over specific subsets of your Trees e.g. sync subset of parameters across devices in a distributed computation.