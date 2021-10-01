<!-- ### Field metadata -->
All metadata you set either with `to.field` or added by Treeo by default will be available in the `field_metadata: Mapping[str, FieldMetadata]` property.

```python
@dataclass
class Person(to.Tree):
    height: float = to.field(node=True)
    name: str = to.field(node=False, opaque=True)

mike = Person(height=1.5, name='Mike')

# no quite true, but you get the idea
assert mike.field_metadata == {
    'height': FieldMetadata(
        node=True, 
        kind=NoneType, 
        opaque=False,
    ), 
    'name': FieldMetadata(
        node=False, 
        kind=None, 
        opaque=True,
    )
}
```

## Changing field metadata
If at anypoint you want to change the metadata of any field you can do so by using the `update_field_metadata` method. For example, imagine we have this definition of BatchNorm: 
```python
class BatchNorm(to.Tree):
    # nodes
    mean: jnp.ndarray = to.field(node=True, kind=BatchStat)
    ...
    # static
    features_in: int
    momentum: float
    ...

model = BatchNorm(features_in=32, momentum=0.9)
```
The `momentum` hyperparameter field is here is a float that you could even wish to make diffentiable in e.g. a meta-gradient setting, however, since the original author of the class didn't consider this its not a node, you can get around this by updating its metadata:

```python
class DifferentiableHyperParam:
    pass

model = model.update_field_metadata(
    'momentum', node=True, kind=DifferentiableHyperParam
)
```