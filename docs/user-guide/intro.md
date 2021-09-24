# User Guide

Before we start it would be useful to define some terminology that will be used throughout the documentation.

### Terminology

* _Type annotations_ are type you set in a class variable after the `:` symbol.
* _Field declarations_ are class variables whose default value is assigned with `field`. Under the hood these are `dataclass.Field` instances.

In code these terms map to the following:

```python
class MyModule(to.Tree):
    #  field      annotation   -----------declaration-------------
    #    v            v        v                                 v
    some_field : jnp.ndarray = to.field(node=True, kind=Parameter)
    #                                      ^                ^
    #                                 node status       field kind
```