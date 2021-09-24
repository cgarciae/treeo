# User Guide

Before we start it would be useful to define some terminology that will be used throughout the documentation.

### Terminology

* **Type Annotation**: ([type hints](https://docs.python.org/3/library/typing.html)) types you set while defining a variable after the `:` symbol.
* **Field Declaration**: default values for class variables that are set using the `field` function.
* **Node Field**: A field that is declared as a node, that is, its content is part of the tree leaves.
* **Static Field**: A field that is declared as a static, that is, its content is not part of the leaves.
* **Field Kind**: An associated type, separate from the type annotation, that gives semantic meaning to the field.

In code these terms map to the following:

```python
class MyModule(to.Tree):
    #  field      annotation   -----------declaration-------------
    #    v            v        v                                 v
    some_field : jnp.ndarray = to.field(node=True, kind=Parameter)
    #                                      ^                ^
    #                                 node status       field kind
```