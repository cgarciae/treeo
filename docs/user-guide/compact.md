# Compact

Treeo's `compact` decorator and `Compact` mixin allow the initialization of fields and the definition of Tree nodes during a function call. `compact` enables a simpler syntax for Trees whose computation structure follows the Tree's structure.

For example, if you have Trees with the following behavior:

```python
import Treeo as to

class Child(to.Tree):
    some_node: float = to.node()

    def __call__(self, x):
        ...
        return x

class Parent(to.Tree):

    def __init__(self):
        self.child1 = Child(10)
        self.child2 = Child(20)
        self.child3 = Child(30)

    def __call__(self, x):
        x = self.child1(x)
        x = self.child2(x)
        x = self.child3(x)
        return x
```

Notice how you have to specify/use the same fields in `__init__` and `__call__`. To reduce the amount of boilerplate you can use the `compact` decorator:

```python
class Parent(to.Tree):
    @to.compact
    def __call__(self, x):
        x = Child(10)(x)
        x = Child(20)(x)
        x = Child(30)(x)
        return x
```

While it seems that `Child` Trees are been created on every call, `compact` will keep track of the Trees created during the first call, assign them as fields to `Parent`, and reuse them on subsequent calls; their constructors will be called only once. 

!!! warning
    You cannot conditionally construct Trees on a compact method unless that conditional doesn't change during the Tree's lifespan. Adding the following to the previous example will cause trouble:

    ```python
    if x.shape[0] > 10:
        x = Child(10)(x)
    ```

    The number and order in which the sub-Trees are defined inside `compact` should always be the same.

## Naming
The names of the created Trees are stored in order of creation in `._subtrees`, the name of the field will be defined as follows:

* If the Tree has a `name` attribute, it will be used as the name of the field.
* Else if it has a `__name__` attribute, it will be used.
* Else a snake_case version of the Tree's class name will be used.
* If a field with the same name already exists, a number will be appended to the name.

The previous example will result in the following fields: `child`, `child2`, `child3`.

## Compact Mixin

With the `Compact` mixin you can add the `get_field` method and the `first_run` property to a Tree subclass. These methods provide mechanisms to initialize fields at runtime potentially based on some properties of the input. As an example let's code a `Linear` Tree that does shape inference for its `w` and `b` parameters:

```python
class Linear(to.Tree, to.Compact):
    w: float = to.node()
    b: float = to.node()

    def __init__(self, dout, key):
        self.dout = dout
        self.key = key

    @to.compact
    def __call__(self, x):
        din = x.shape[-1]
        w = self.get_field("w", lambda: jax.random.uniform(self.key, [din, self.dout]))
        b = self.get_field("b", lambda: jnp.zeros(shape=[self.dout]))

        return jnp.dot(x, w) + b
```

`get_field` will initialize the `w` and `b` fields on the first run and fetch their values on subsequent runs. You can also use the `first_run` property and manually initialize the fields:

    
```python
class Linear(to.Tree, to.Compact):
    w: float = to.node()
    b: float = to.node()

    def __init__(self, dout, key):
        self.dout = dout
        self.key = key

    @to.compact
    def __call__(self, x):
        if self.first_run:
            din = x.shape[-1]
            self.w = jax.random.uniform(self.key, [din, self.dout])
            self.b = jnp.zeros(shape=[self.dout])

        return jnp.dot(x, self.w) + self.b
```

This is useful if you want to perform more complex initialization procedures.
