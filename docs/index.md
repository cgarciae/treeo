# Treeo

_A small library for creating and manipulating custom JAX Pytree classes_

* **Light-weight**: has no dependencies other than `jax`.
* **Compatible**: Treeo `Tree` objects are compatible with any `jax` function that accepts Pytrees.
* **Standards-based**: `treeo.field` is built on top of python's `dataclasses.field`.
* **Flexible**: Treeo is compatible with both dataclass and non-dataclass classes.

Treeo lets you easily create class-based Pytrees so your custom objects can easily interact seamlessly with JAX. Uses of Treeo can range from just creating simple simple JAX-aware utility classes to using it as the core abstraction for full-blown frameworks. Treeo was originally extracted from the core of [Treex](https://github.com/cgarciae/treex) and shares a lot in common with [flax.struct](https://flax.readthedocs.io/en/latest/flax.struct.html#module-flax.struct).

[Documentation](https://cgarciae.github.io/treeo) | [User Guide](https://cgarciae.github.io/treeo/user-guide/intro)

## Installation
Install using pip:
```bash
pip install treeo
```

## Basics
With Treeo you can easily define your own custom Pytree classes by inheriting from Treeo's `Tree` class and using the `field` function to declare which fields are nodes (children) and which are static (metadata):

```python
import treeo as to

@dataclass
class Person(to.Tree):
    height: jnp.array = to.field(node=True) # I am a node field!
    name: str = to.field(node=False) # I am a static field!
```
`field` is just a wrapper around `dataclasses.field` so you can define your Pytrees as dataclasses, but Treeo fully supports non-dataclass classes as well. Since all `Tree` instances are Pytree they work with the various functions from the`jax` library as expected:

```python
p = Person(height=jnp.array(1.8), name="John")

# Trees can be jitted!
jax.jit(lambda person: person)(p) # Person(height=array(1.8), name='John')

# Trees can be mapped!
jax.tree_map(lambda x: 2 * x, p) # Person(height=array(3.6), name='John')
```
#### Kinds
Treeo also include a kind system that lets you give semantic meaning to fields (what a field represents within your application). A kind is just a type you pass to `field` via its `kind` argument: 

```python
class Parameter: pass
class BatchStat: pass

class BatchNorm(to.Tree):
    scale: jnp.ndarray = to.field(node=True, kind=Parameter)
    mean: jnp.ndarray = to.field(node=True, kind=BatchStat)
```

Kinds are very useful as a filtering mechanism via [treeo.filter](https://cgarciae.github.io/treeo/user-guide/api/filter):

```python 
model = BatchNorm(...)

# select only Parameters, mean is filtered out
params = to.filter(model, Parameter) # BatchNorm(scale=array(...), mean=Nothing)
```
`Nothing` behaves like `None` in Python, but it is a special value that is used to represent the absence of a value within Treeo.

Treeo also offers the [merge](https://cgarciae.github.io/treeo/user-guide/api/merge) function which lets you rejoin filtered Trees with a logic similar to Python `dict.update` but done recursively:
```python hl_lines="3"
def loss_fn(params, model, ...):
    # add traced params to model
    model = to.merge(model, params)
    ...

# gradient only w.r.t. params
params = to.filter(model, Parameter) # BatchNorm(scale=array(...), mean=Nothing)
grads = jax.grad(loss_fn)(params, model, ...)
```

For a more in-depth tour check out the [User Guide](https://cgarciae.github.io/treeo/user-guide/intro).

## Examples

### A simple Tree
```python
from dataclasses import dataclass
import treeo as to

@dataclass
class Character(to.Tree):
    position: jnp.ndarray = to.field(node=True)    # node field
    name: str = to.field(node=False, opaque=True)  # static field

character = Character(position=jnp.array([0, 0]), name='Adam')

# character can freely pass through jit
@jax.jit
def update(character: Character, velocity, dt) -> Character:
    character.position += velocity * dt
    return character

character = update(character velocity=jnp.array([1.0, 0.2]), dt=0.1)
```
### A Stateful Tree
```python
from dataclasses import dataclass
import treeo as to

@dataclass
class Counter(to.Tree):
    n: jnp.array = to.field(default=jnp.array(0), node=True) # node
    step: int = to.field(default=1, node=False) # static

    def inc(self):
        self.n += self.step

counter = Counter(step=2) # Counter(n=jnp.array(0), step=2)

@jax.jit
def update(counter: Counter):
    counter.inc()
    return counter

counter = update(counter) # Counter(n=jnp.array(2), step=2)

# map over the tree
```

### Full Example - Linear Regression

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
