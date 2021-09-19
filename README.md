# Pytreex

_A library for easily creating and manipulating custom JAX Pytrees_

* **Light-weight**: has dependencies other than `jax`.
* **Compatible**: Pytreex `Tree` objects should be compatible with any `jax` function that accepts Pytrees.
* **Pythonic**: `treeo.field` is built on top of python's `dataclasses.field` so `treeo.Tree` subclasses can be defined as both dataclass and and non-dataclass classes.


[Documentation](https://cgarciae.github.io/treeo) | [Guide](#guide)

## Installation
Install using pip:
```bash
pip install treeo
```

## Getting Started
This is a small appetizer to give you a feel for how using Pytreex looks like, be sure to checkout the [Guide section](#guide) below for details on more advanced usage.
```python
from dataclasses import dataclass
import jax.numpy as jnp
import treeo as to

@dataclass
class Character(to.Tree):
    position: jnp.ndarray = to.field(node=True)  # node field
    velocity: jnp.ndarray = to.field(node=True)  # node field
    id: str = to.field(node=False)  # static field

    def move(self, dt: float):
        self.position += self.velocity * dt

character = Character(
    position=jnp.array([0, 0]),
    velocity=jnp.array([1, 1]),
    id='1',
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
You can define a `kind` for a field, this is useful for for filtering field value using the [treeo.filter](#filter) function. A `kind` is can be any `type` e.g.

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

### Changing node status

### Sugar

### API
#### Filter
The `filter` method allows you to select a subtree by filtering based on a `TreePart` type, all leaves whose type annotations are a subclass of such type are kept, the rest are set to a special `Nothing` value.
```python
class MyModule(to.Tree):
    a: to.Parameter[np.ndarray] = np.array(1)
    b: to.BatchStat[np.ndarray] = np.array(2)
    ...

module = MyModule(...)

module.filter(to.Parameter) # MyModule(a=array([1]), b=Nothing)
module.filter(to.BatchStat) # MyModule(a=Nothing, b=array([2]))
```
Since `Nothing` is an empty Pytree it gets ignored by tree operations, this effectively allows you to easily operate on a subset of the fields:

```python
jax.tree_map(lambda x: -x, module.filter(to.Parameter)) # MyModule(a=array([-1]), b=Nothing)
jax.tree_map(lambda x: -x, module.filter(to.BatchStat)) # MyModule(a=Nothing, b=array([-2]))
```

##### filter predicates
If you need to do more complex filtering, you can pass callables with the signature `FieldInfo -> bool` instead of types:

```python
# all States that are not OptStates
module.filter(
    lambda field: issubclass(field.annotation, to.State) 
    and not issubclass(field.annotation, to.OptState)
) 
# MyModule(a=Nothing, b=array([2]))
```
##### multiple filters
The previous could be abbreviated using multiple filters as its required that **all** filters pass for a field to be kept:
```python
# all States that are not OptStates
module.filter(
    to.State,
    lambda field: not issubclass(field.annotation, to.OptState)
) 
# MyModule(a=Nothing, b=array([2]))
```
The previous also be written as:
```python
module.states(lambda field: not issubclass(field.annotation, to.OptState))
```

#### Update
The `update` method allows you to merge the values of one or more incoming modules with the current module, this is useful for integrating filtered modules back into the main module.

```python
module = MyModule(...) # MyModule(a=array([1]), b=array([2]))
params = module.parameters() # MyModule(a=array([1]), b=Nothing)
negative = jax.tree_map(lambda x: -x, params) # MyModule(a=array([-1]), b=Nothing)
module = module.update(negative) # MyModule(a=array([-1]), b=array([2]))
```

#### Map
The `map` method provides a convenient way to map a function over the fields of a module:

```python
module = MyModule(...) # MyModule(a=array([1]), b=array([2]))
params = module.parameters() # MyModule(a=array([1]), b=Nothing)
negative = params.map(lambda x: -x) # MyModule(a=array([-1]), b=Nothing)
module = module.update(negative) # MyModule(a=array([-1]), b=array([2]))
```

The previous pattern is so common that `map` provides a shortcut for applying `filter -> tree_map -> update` in sequence:

```python
module = MyModule(...) # MyModule(a=array([1]), b=array([2]))
module = module.map(lambda x: -x, to.Parameter) # MyModule(a=array([-1]), b=array([2]))
```

As shown here, `map` accepts the same `*args` as `filter` and calls `update` at the end if filters are given.

### State Management
Pytreex takes a "direct" approach to state management, i.e., state is updated in-place by the Tree whenever it needs to. For example, this module will calculate the running average of its input:
```python
class Average(to.Tree):
    count: to.State[jnp.ndarray]
    total: to.State[jnp.ndarray]

    def __init__(self):
        super().__init__()
        self.count = jnp.array(0)
        self.total = jnp.array(0.0)

    def __call__(self, x):
        self.count += np.prod(x.shape)
        self.total += jnp.sum(x)

        return self.total / self.count
```
Pytreex Pytrees that require random state will often keep a `rng` key internally and update it in-place when needed:
```python
class Dropout(to.Tree):
    rng: to.Rng[to.Initializer, jnp.ndarray]  # Initializer | ndarray

    def __init__(self, rate: float):
        ...
        self.rng = to.Initializer(lambda key: key)
        ...

    def __call__(self, x):
        key, self.rng = jax.random.split(self.rng)
        ...
```
Finally `to.Optimizer` also performs inplace updates inside the `apply_updates` method, here is a sketch of how it works:
```python
class Optimizer(to.TreeObject):
    opt_state: to.OptState[Any]
    optimizer: optax.GradientTransformation

    def apply_updates(self, grads, params):
        ...
        updates, self.opt_state = self.optimizer.update(
            grads, self.opt_state, params
        )
        ...
```

#### What is the catch?
<!-- TODO: Add a list of rules to follow around jitted functions -->
State management is one of the most challenging things in JAX, but with the help of Pytreex it seems effortless, what is the catch? As always there is a trade-off to consider: Pytreex's approach requires to consider how to propagate state changes properly while taking into account the fact that Pytree operations create new objects, that is, since reference do not persist across calls through these functions changes might be lost. 

A standard solution to this problem is: **always output the module to update state**. For example, a typical loss function that contains a stateful model would look like this:

```python
@partial(jax.value_and_grad, has_aux=True)
def loss_fn(params, model, x, y):
    model = model.update(params)

    y_pred = model(x)
    loss = jnp.mean((y_pred - y) ** 2)

    return loss, model

params = model.parameters()
(loss, model), grads = loss_fn(params, model, x, y)
...
```
Here `model` is returned along with the loss through `value_and_grad` to update `model` on the outside thus persisting any changes to the state performed on the inside.

### Non-hashable static fields
If you want to have a static field that contains a non-hashable value like a numpy or jax array, you can use `to.Hashable` to wrap around it such that it:

```python
class MyModule(to.Tree):
    table: to.Hashable[np.ndarray]
    ...

    def __init__(self, table: np.ndarray):
        self.table = to.Hashable(table)
        ...
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        table = self.table.value
        ...
```
The hash from `Hashable` will only depend on object identity but not on the actual `value`, therefore you should treat it as an immutable, if you want to update its value you should create a new `Hashable` instance:

```python
table = np.ones((10, 10))
module = MyModule(table)

# use module as an argument for a jit-ed function
...

module.table = to.Hashable(np.zeros((10, 10)))

# jit-ed function will recompile now
...
```
If you are somehow able to mutate `value` directly JAX won't know about this and `jit` won't recompile.

**Note:** Currently JAX does not complain when you have a static field is a numpy array, however, in case you mutate such field and pass its module through jit again you will get a deprecation warning saying this situation will be an error in the future.

### Full Example

```python
from functools import partial
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import treeo as to

x = np.random.uniform(size=(500, 1))
y = 1.4 * x - 0.3 + np.random.normal(scale=0.1, size=(500, 1))

# treeo already defines to.Linear but we can define our own
class Linear(to.Tree):
    w: to.Parameter[to.Initializer, jnp.ndarray]
    b: to.Parameter[jnp.ndarray]

    def __init__(self, din, dout):
        super().__init__()
        self.w = to.Initializer(lambda key: jax.random.uniform(key, shape=(din, dout)))
        self.b = jnp.zeros(shape=(dout,))

    def __call__(self, x):
        return jnp.dot(x, self.w) + self.b


model = Linear(1, 1).init(42)
optimizer = to.Optimizer(optax.adam(0.01))
optimizer = optimizer.init(model.paramerters())


@partial(jax.value_and_grad, has_aux=True)
def loss_fn(params, model, x, y):
    model = model.update(params)

    y_pred = model(x)
    loss = jnp.mean((y_pred - y) ** 2)

    return loss, model


@jax.jit
def train_step(model, x, y, optimizer):
    params = model.paramerters()
    (loss, model), grads = loss_fn(params, model, x, y)

    # here model == params
    model = optimizer.apply_updates(grads, model)

    return loss, model, optimizer


for step in range(1000):
    loss, model, optimizer = train_step(model, x, y, optimizer)
    if step % 100 == 0:
        print(f"loss: {loss:.4f}")

model = model.eval()

X_test = np.linspace(x.min(), x.max(), 100)[:, None]
y_pred = model(X_test)

plt.scatter(x, y, c="k", label="data")
plt.plot(X_test, y_pred, c="b", linewidth=2, label="prediction")
plt.legend()
plt.show()
```
