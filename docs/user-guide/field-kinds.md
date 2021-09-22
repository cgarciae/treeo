<!-- ### Field kinds -->
You can define a `kind` for a field, this is useful for for filtering field value using the [treeo.filter](#filter) function. A `kind` is can be any `type`, it is only there fore metadata that `filter` can leverage. For example, here is a possible definition for a `BatchNorm` module using `kind`s:

```python
import treeo as to

class Parameter:
    pass

class BatchStat:
    pass

class BatchNorm(to.Tree):
    scale: jnp.ndarray = to.field(node=True, kind=Parameter)
    bias: jnp.ndarray = to.field(node=True, kind=Parameter)

    mean: jnp.ndarray = to.field(node=True, kind=BatchStat)
    var: jnp.ndarray = to.field(node=True, kind=BatchStat)
    ...
```
Based on this definition, `filter` can query specific kind of fields:

```python
model = BatchNorm(...) 

# BatchNorm(scale=array(...), bias=array(...), mean=Nothing, var=Nothing)
params = to.filter(model, Parameter) # filter by kind

# BatchNorm(scale=Nothing, bias=Nothing, mean=array(...), var=array(...))
batch_stats = to.filter(model, BatchStat)
```