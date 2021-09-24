# Treeo

_A small library for creating and manipulating custom JAX Pytree classes_

* **Light-weight**: has no dependencies other than `jax`.
* **Compatible**: Treeo `Tree` objects are compatible with any `jax` function that accepts Pytrees.
* **Standards-based**: `treeo.field` is built on top of python's `dataclasses.field`.
* **Flexible**: Treeo is compatible with both dataclass and non-dataclass classes.

Treeo was originally extracted from the core of [Treex](https://github.com/cgarciae/treex) and (although the author was not aware of this at the time) shares a lot in common with [flax.struct](https://flax.readthedocs.io/en/latest/flax.struct.html#module-flax.struct). Treeo has nothing in particular to do with Deep Learning, but some of the examples are motivated by it.

[Documentation](https://cgarciae.github.io/treeo) | [Guide](https://cgarciae.github.io/treeo/user-guide/intro)

## Installation
Install using pip:
```bash
pip install treeo
```

## Basics
At its core Treeo focuses on 2 things:

* Tooling for defining pytree structures via field metadata.
* A set of functions for manipulating pytree structures that leverage the field metadata.

As part of the field metadata Treeo introduce the concept of a `kind` which enables a powerful filtering mechanism.

#### Fields
To define node fields for a custom Pytree, Treeo uses the `field` function which is a wrapper around `dataclasses.field`:

```python
import treeo as to

@dataclass
class Person(to.Tree):
    height: jnp.array = to.field(node=True) # I am a node field!
    name: str = to.field(node=False) # I am a static field!
```

Using this information Treeo specifies a Pytree you can use with various `jax` functions.

```python
p = Person(height=jnp.array(1.8), name="John")

# Trees can be jitted!
jax.jit(lambda person: person)(p) # Person(height=array(1.8), name='John')

# Trees can be mapped!
jax.tree_map(lambda x: 2 * x, p) # Person(height=array(3.6), name='John')
```
#### Kinds
Kinds allow you to imbue semantic information into your Pytree, this is critcal for filtering operations where cannot know the purpose of a leaf based on its type. Using the `kind` argument of `field` and Treeo's `filter` function you select only the nodes you want to keep:

```python hl_lines="10"
class Parameter: pass
class BatchStat: pass

class BatchNorm(to.Tree):
    scale: jnp.ndarray = to.field(node=True, kind=Parameter)
    mean: jnp.ndarray = to.field(node=True, kind=BatchStat)

def loss_fn(params, model, ...):
    # merge params back into model
    model = to.update(model, params)
    ...

model = BatchNorm(...)

# select only Parameters, mean is filtered out
params = to.filter(model, Parameter) # BatchNorm(scale=array(...), mean=Nothing)

grads = jax.grad(loss_fn)(params, model, ...)
```

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
    character.position += character.velocity * dt
    return character

character = update(character velocity=jnp.array([1.0, 0.2]), dt=0.1)
```
### Stateful Trees
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

def update(counter: Counter):
    counter.inc()
    return counter

counter = update(counter) # Counter(n=jnp.array(2), step=2)

# map over the tree
```

### Full Example - Linear Regression

```python
import matplotlib.pyplot as plt
import numpy as np
import treeo as to


class Parameter:
    pass

class Linear(to.Tree):
    din: int # static
    dout: int # static

    w: jnp.ndarray = to.field(node=True, kind=Parameter)
    b: jnp.ndarray = to.field(node=True, kind=Parameter)

    def __init__(self, din, dout, key):
        self.din = din
        self.dout = dout
        self.b = jnp.zeros(shape=(dout,))

    def __call__(self, x):
        return jnp.dot(x, self.w) + self.b


def loss_fn(model, x, y):

    y_pred = model(x)
    loss = jnp.mean((y_pred - y) ** 2)

    return loss


def sgd(param, grad):
    return param - 0.1 * grad


def train_step(model, x, y):
    loss, grads = loss_fn(model, x, y)


    return loss, model


x = np.random.uniform(size=(500, 1))
y = 1.4 * x - 0.3 + np.random.normal(scale=0.1, size=(500, 1))

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
