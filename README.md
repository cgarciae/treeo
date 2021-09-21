# Treeo

_A small library for creating and manipulating custom JAX Pytree classes_

* **Light-weight**: has no dependencies other than `jax`.
* **Compatible**: Treeo `Tree` objects are compatible with any `jax` function that accepts Pytrees.
* **Standards-based**: `treeo.field` is built on top of python's `dataclasses.field`
* **Flexible**: Treeo is compatible with both dataclass and non-dataclass classes.

Treeo was originally extracted from the core of [Treex](https://github.com/cgarciae/treex) and (although the author was not aware of this at the time) shares a lot in common with [flax.struct](https://flax.readthedocs.io/en/latest/flax.struct.html#module-flax.struct). Treeo has nothing in particular to do with Deep Learning, but some of the examples are motivated by it.

[Documentation](https://cgarciae.github.io/treeo) | [Guide](#guide)

## Installation
Install using pip:
```bash
pip install treeo
```

## Getting Started
This is a small appetizer to give you a feel for how using Treeo looks like, be sure to checkout the [Guide section](#guide) below for details on more advanced usage.
```python
from dataclasses import dataclass
import jax.numpy as jnp
import treeo as to

@dataclass
class Character(to.Tree):
    position: jnp.ndarray = to.field(node=True)    # node field
    velocity: jnp.ndarray = to.field(node=True)    # node field
    name: str = to.field(node=False, opaque=True)  # static field

    def move(self, dt: float):
        self.position += self.velocity * dt

character = Character(
    position=jnp.array([0, 0]),
    velocity=jnp.array([1, 1]),
    name='1',
)

# character can freely pass through jit
@jax.jit
def update(character: Character, dt: float):
    character.move(dt)
    return character

character = update(character, 0.1)
```

## Guide

### Tree fields
Tree fields are divided into two categories:
* `node` fields: they are considered as part of the pytree, JAX functions such as `tree_map` will operate over them.
* `static` fields: they are part of the `PyTreeDef`, JAX functions will not operate over them, but JAX is still aware of them, e.g. JAX will recompile jitted functions is case these fields change.
```python
@dataclass
class Person(to.Tree):
    height: float = to.field(node=True)
    name: str = to.field(node=False)

person = Person(height=1.5, name='John')

tree_map(lambda x: x + 1, person) # Person(height=2.5, name='John')
```
Since `to.field` is just a wrapper over `dataclasses.field` that adds the `node` and `kind` arguments you can use all `dataclass` features. However, dataclasses are orthogonal to Treeo, this means that you can naturally use non-dataclass classes:

```python
class Person(to.Tree):
    height: float = to.field(node=True)
    name: str = to.field(node=False)

    def __init__(self, height: float, name: str):
        self.height = height
        self.name = name

person = Person(height=1.5, name='John')

tree_map(lambda x: x + 1, person) # Person(height=2.5, name='John')
```

### Field kinds
You can define a `kind` for a field, this is useful for for filtering field value using the [treeo.filter](#filter) function. A `kind` is can be any `type`, it is only there fore metadata that `filter` can leverage. For example, here is a possible definition for a `BatchNorm` module using `kind`s:

```python
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

### Node policy
If a field is **not** marked with `to.field` the following policy will be applied when determining whether a field is a node or not:
* If the field is _annotated_ with a `Tree` subtype or a generic containing a `Tree` subtype e.g. `List[to.Tree]`, the field is considered a node.
* If the runtime value of the field is a `to.Tree` instance, the field is considered a node and the `Tree`'s metadata will be updated to reflect this.
* If none of the above apply, the field is considered a static field.

```python
class Agent(to.Tree):
    ...

class Game(to.Tree):
    player: Agent      # node
    cpus: List[Agent]  # node
    max_cpus: int      # static

    def __init__(self, ...):
        ...
        self.boss = Agent(...) # runtime node
```

**Note:** the `kind` of all fields that are not explicitly declarated is set to `NoneType`.

### Opaque static fields
When the value of a static field changes `jit` and friends will recompile to reflect posible changes in the computation logic. While this is good quality in many cases, sometimes certain fields are just there for metadata and should not part of the actual computation. For example, `name` in the `Person` example above will probably not affect the logic yet in this example it will force recompilation:


```python
@dataclass
class Person(to.Tree):
    height: float = to.field(node=True)
    name: str = to.field(node=False)

@jax.jit
def do_something(person: Person):
    ...

do_something(Person(height=1.5, name='John')) # compiles
do_something(Person(height=1.5, name='Fred')) # re-compiles 🙁
```
To avoid this, you can mark a field as `opaque`:
```python
@dataclass
class Person(to.Tree):
    height: float = to.field(node=True)
    name: str = to.field(node=False, opaque=True)

@jax.jit
def do_something(person: Person):
    ...

do_something(Person(height=1.5, name='John')) # compiles
do_something(Person(height=1.5, name='Fred')) # cached! 🤩
```
`opaque` will "hide" the value content of the field from JAX, changes will only be detected if the type of an opaque field changes or, in case its an array-like type, if its shape or dtype changes.

#### opaque_is_equal
If you want to define you on policy on how opaque fields are handled, you can use the `opaque_is_equal: Callable[[to.Opaque, Any], bool]` argument and pass a function that takes in a `to.Opaque(value: Any)` object and the new value of the field and return a boolean indicating whether the new value is considered equal to the opaque value.

```python
def same_length(opaque: to.Opaque, other: Any):
    if (
        isinstance(other, to.Opaque) 
        and type(opaque.value) == type(other.value)
    ):
        if isinstance(opaque.value, list):
            return len(opaque.value) == len(other.value)
        else:
            return True
    else:
        return False

@dataclass
class Person(to.Tree):
    height: float = to.field(node=True)
    names: List[str] = to.field(
        node=False, opaque=True, opaque_is_equal=same_length
    )
```
### Field metadata
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
        opaque_is_equal=None,
    ), 
    'name': FieldMetadata(
        node=False, 
        kind=None, 
        opaque=True,
        opaque_is_equal=None,
    )
}
```

#### Changing field metadata
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
### Sugar 🍬
For pedagogical reason, `to.field` has been used throught the documentation to reinforce the concepts, however, Treeo has a couple of shortcuts to make it easier and more understandable to define Trees.

| normal                                  | shortcut              |
| --------------------------------------- | --------------------- |
| `to.field(node=True)`                   | `to.node()`           |
| `to.field(node=False)`                  | `to.static()`         |
| `to.field(node=True, kind=TreeOrKind)`  | `TreeOrKind.node()`   |
| `to.field(node=False, kind=TreeOrKind)` | `TreeOrKind.static()` |

Based on this, you can take the following code

```python
class Parameter:
    pass

class Child(to.Tree):
    a: float = to.field(node=True)
    b: str = to.field(node=False)
    b: float = to.field(node=True, kind=Parameter)
    d: float = to.field(node=False, kind=Parameter)

class Parent(to.Tree):
    child1: Child = to.field(node=True, kind=Child)
    child2: Child = to.field(node=False, kind=Child)
    rest: List[Child] = to.field(node=True, kind=Child)
```
and simplify it to:
```python
class Parameter(to.KindMixin):
    pass

class Child(to.Tree):
    a: float = to.node()
    b: str # = to.static(), inferred
    b: float = Parameter.node()
    d: float = Parameter.static()

class Parent(to.Tree):
    child1: Child # = Child.node(), inferred
    child2: Child = Child.static()
    rest: List[Child] # = Child.node(), inferred
```
As you see we use the `to.KindMixin` to add some methods to the `Parameter` class. Also, since the `Tree` already inherits from `to.KindMixin`, we can use the `.node()` and `.static()` methods on the `Tree` subclass.

### API
Throught these examples for the functional API we will use the following defintions:
```python
class Parameter(to.KindMixin):
    pass

class BatchStat(to.KindMixin):
    pass

@dataclass
class MyTree(to.Tree):
    a: jnp.ndarray = Parameter.node()
    b: jnp.ndarray = BatchStat.node()
```
#### Filter
The `to.filter` function allows you to select a subtree by filtering based on a `kind`, all leaves whose field kind is a subclass of such type are kept, the rest are set to a special `Nothing` value.
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

##### filter predicates
If you need to do more complex filtering, you can pass callables with the signature `FieldInfo -> bool` instead of types:

```python
# all Parameters whose field name is "kernel"
to.filter(
    tree,
    lambda field: issubclass(field.kind, State) 
    and field.name == "kernel"
) 
# MyTree(a=Nothing, b=Nothing)
```
##### multiple filters
The previous could be abbreviated using multiple filters as its required that **all** filters pass for a field to be kept, since passing types by themselves filters by kind, the previous example could be written as:
```python
# all Parameters whose field name is "kernel"
to.filter(
    tree,
    Parameter,
    lambda field: field.name == "kernel"
) 
# MyTree(a=Nothing, b=Nothing)
```

#### Update
The `to.update` function allows you to merge the values of one or more incoming pytrees with the current pytree, this is useful for integrating filtered Trees back into the main Tree.

```python
tree = MyTree(a=jnp.array(1), b=jnp.array(2))
params = to.filter(tree, Parameter) # MyTree(a=array(1), b=Nothing)
negative = jax.tree_map(lambda x: -x, params) # MyTree(a=array(-1), b=Nothing)
tree = to.update(tree, negative) # MyTree(a=array(-1), b=array(2))
```

#### Map
The `map` function provides a convenient way to map a function over the fields of a pytree:

```python
tree = MyTree(a=jnp.array(1), b=jnp.array(2))
params = to.filter(tree, Parameter) # MyTree(a=array(1), b=Nothing)
negative = to.map(lambda x: -x, params) # MyTree(a=array(-1), b=Nothing)
tree = to.update(tree, negative) # MyTree(a=array(-1), b=array(2))
```

Up to this point `to.map` is behaving just like `jax.tree_map`, however, the pattern in the previous example is so common that `map`s main use is providing a shortcut for applying `filter -> tree_map -> update` in sequence:

```python
tree = MyTree(a=jnp.array(1), b=jnp.array(2))
tree = to.map(lambda x: -x, tree, Parameter) # MyTree(a=array(-1), b=array(2))
```

As shown here, `map` accepts the same `*args` as `filter` and calls `update` at the end if filters are given.

### State Management
Treeo takes a "direct" approach to state management, i.e., state is updated in-place by the Tree whenever it needs to. For example, this module will calculate the running average of its input:
```python
@dataclass
class Average(to.Tree):
    count: State[jnp.ndarray] = jnp.array(0)
    total: State[jnp.ndarray] = jnp.array(0.0)

    def __call__(self, x):
        self.count += np.prod(x.shape)
        self.total += jnp.sum(x)

        return self.total / self.count
```


#### What is the catch?
<!-- TODO: Add a list of rules to follow around jitted functions -->
State management is one of the most challenging things in JAX, but with the help of Treeo it seems effortless, what is the catch? As always there is a trade-off to consider: Treeo's approach requires to consider how to propagate state changes properly while taking into account the fact that Pytree operations create new objects, that is, since reference do not persist across calls through these functions changes might be lost. 

A standard solution to this problem is: **always output the Tree to update state**. For example, a typical gradient function in a Deep Learning application that contains a stateful Tree would look like this:

```python
@partial(jax.value_and_grad, has_aux=True)
def grad_fn(params, model, x, y):
    model = to.update(model, params)

    y_pred = model(x) # state is updated
    loss = jnp.mean((y_pred - y) ** 2)

    return loss, model # return model to propagate state changes

params = to.filter(model, Parameter)
(loss, model), grads = grad_fn(params, model, x, y)
...
```
Here `model` is returned along with the loss through `value_and_grad` to update `model` on the outside thus persisting any changes to the state performed on the inside.

### Non-hashable static fields
Static fields are required to be hashable by JAX, so what happens if you want to have a static field that contains a non-hashable value like a numpy or jax array? For example:

```python
```python
@dataclass
class MyTree(to.Tree):
    table: np.ndarray
```

One solution is to make that field `opaque`, this will work as long as you don't need `jit` and friends to reach to changes to its value content for recompiling.  If you do, you can instead use the `to.Hashable(value)` class to wrap around it like this:

```python
@dataclass
class MyTree(to.Tree):
    table: to.Hashable[np.ndarray]
```

The hash from `Hashable` will only depend on object identity but not on the actual `value`, therefore you should treat it as immutable, if you want to update its content you should create a new `Hashable` instance:

```python
table = to.Hasable(np.ones((10, 10)))
tree = MyTree(table)

@jax.jit
def do_something(tree: MyTree):
    table_value = tree.table.value
    ...

tree = do_something(tree) # compiles
tree = do_something(tree) # uses cache

# update table
module.table = to.Hashable(np.zeros((10, 10)))

tree = do_something(tree) # recompiles
```
**Warning**: If you are somehow able to mutate `Hashable.value` directly JAX won't know about this and `jit` won't recompile.

### Full Example

```python
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import treeo as to


class Linear(to.Tree):
    w: jnp.ndarray = to.node()
    b: jnp.ndarray = to.node()

    def __init__(self, din, dout, key):
        self.w = jax.random.uniform(key, shape=(din, dout))
        self.b = jnp.zeros(shape=(dout,))

    def __call__(self, x):
        return jnp.dot(x, self.w) + self.b


@jax.value_and_grad
def loss_fn(model, x, y):

    y_pred = model(x)
    loss = jnp.mean((y_pred - y) ** 2)

    return loss


def sgd(param, grad):
    return param - 0.1 * grad


@jax.jit
def train_step(model, x, y):
    loss, grads = loss_fn(model, x, y)

    model = jax.tree_map(sgd, model, grads)

    return loss, model


x = np.random.uniform(size=(500, 1))
y = 1.4 * x - 0.3 + np.random.normal(scale=0.1, size=(500, 1))

key = jax.random.PRNGKey(0)
model = Linear(1, 1, key=key)

for step in range(1000):
    loss, model = train_step(model, x, y)
    if step % 100 == 0:
        print(f"loss: {loss:.4f}")

X_test = np.linspace(x.min(), x.max(), 100)[:, None]
y_pred = model(X_test)

plt.scatter(x, y, c="k", label="data")
plt.plot(X_test, y_pred, c="b", linewidth=2, label="prediction")
plt.legend()
plt.show()
```
