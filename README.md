# Pytreex

_A JAX library to easily create and manipulate custom Pytrees_


[Documentation](https://cgarciae.github.io/pytreex) | [Guide](#guide)

## Installation
Install using pip:
```bash
pip install pytreex
```

## Getting Started
This is a small appetizer to give you a feel for how using Pytreex looks like, be sure to checkout the [Guide section](#guide) below for details on more advanced usage.
```python
import jax.numpy as jnp
import pytreex as ptx

class Character(ptx.Tree):
    position: jnp.ndarray = ptx.node()
    velocity: jnp.ndarray = ptx.node()
    id: str = ptx.static()

    def __init__(
        self,
        position: jnp.ndarray,
        velocity: jnp.ndarray,
        id: str,
    ):
        super().__init__()
        self.position = position
        self.velocity = velocity
        self.id = id

    def move(self, dt: float):
        self.position += self.velocity * dt

character = Character('1', jnp.array([0, 0]), jnp.array([1, 1]))

@jax.jit
def update(character: Character, dt: float):
    character.move(dt)
    return character

character = update(character, 0.1)
```

## Guide
Basics

```python
class Character(ptx.Tree):
    position: jnp.ndarray = ptx.field(node=True)
    velocity: jnp.ndarray = ptx.field(node=True)
    id: str = ptx.field(static=False)
```

```python
character = Character(...)

# double position and velocity
character = jax.tree_map(lambda x: 2.0 * x, character)

# path through jit
@jax.jit
def update(character: Character, ...):
    # do stuff
    ...
    return character

character = update(character, ...)
```

```python
class Character(ptx.Tree):
    position: jnp.ndarray = ptx.node()
    velocity: jnp.ndarray = ptx.node()
    id: str = ptx.static()
```

```python
class Position:
    pass

class Velocity:
    pass

class Character(ptx.Tree):
    position: jnp.ndarray = ptx.node(kind=Position)
    velocity: jnp.ndarray = ptx.node(kind=Velocity)
    id: str = ptx.static()
    ...

characters = [Character(...), Character(...), ...]
...
# reset position
positions = ptx.filter(characters, Position) # filter Positions
positions = jax.tree_map(jnp.zeros_like, positions) # reset
character = ptx.update(characters, positions) #

# alternative we could just use `ptx.map`
characters = ptx.map(jnp.zeros_like, characters, Position)
```

```python
class Position(ptx.FieldMixin):
    pass

class Velocity(ptx.FieldMixin):
    pass

class Character(ptx.Tree):
    position: jnp.ndarray = Position.node()
    velocity: jnp.ndarray = Velocity.node()
    id: str = ptx.static()
```

```python
class Game(ptx.Tree):
    cpus: List[Character] = ptx.node(kind=Character)
    player: Character = ptx.node(kind=Character)
    player_checkpoint: Character = ptx.static(kind=Character)
```

```python
class Game(ptx.Tree):
    cpus: List[Character] = Character.node()
    player: Character = Character.node()
    player_checkpoint: Character = Character.static()
```

```python
class Game(ptx.Tree):
    cpus: List[Character]
    player: Character
    player_checkpoint: Character = Character.static()
```

```python
class Game(ptx.Tree):
    cpus = Character.node()
    player_checkpoint = Character.static()

    def __init__(...):
        super().__init__()
        self.player = Character(...)
        ...
```

```python
game = Game(...)

# change `player_checkpoint` from static to node
game = game.update_field_metadata("player_checkpoint", node=True)
```

### API
#### Filter
The `filter` method allows you to select a subtree by filtering based on a `TreePart` type, all leaves whose type annotations are a subclass of such type are kept, the rest are set to a special `Nothing` value.
```python
class MyModule(ptx.Tree):
    a: ptx.Parameter[np.ndarray] = np.array(1)
    b: ptx.BatchStat[np.ndarray] = np.array(2)
    ...

module = MyModule(...)

module.filter(ptx.Parameter) # MyModule(a=array([1]), b=Nothing)
module.filter(ptx.BatchStat) # MyModule(a=Nothing, b=array([2]))
```
Since `Nothing` is an empty Pytree it gets ignored by tree operations, this effectively allows you to easily operate on a subset of the fields:

```python
jax.tree_map(lambda x: -x, module.filter(ptx.Parameter)) # MyModule(a=array([-1]), b=Nothing)
jax.tree_map(lambda x: -x, module.filter(ptx.BatchStat)) # MyModule(a=Nothing, b=array([-2]))
```

##### filter predicates
If you need to do more complex filtering, you can pass callables with the signature `FieldInfo -> bool` instead of types:

```python
# all States that are not OptStates
module.filter(
    lambda field: issubclass(field.annotation, ptx.State) 
    and not issubclass(field.annotation, ptx.OptState)
) 
# MyModule(a=Nothing, b=array([2]))
```
##### multiple filters
The previous could be abbreviated using multiple filters as its required that **all** filters pass for a field to be kept:
```python
# all States that are not OptStates
module.filter(
    ptx.State,
    lambda field: not issubclass(field.annotation, ptx.OptState)
) 
# MyModule(a=Nothing, b=array([2]))
```
The previous also be written as:
```python
module.states(lambda field: not issubclass(field.annotation, ptx.OptState))
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
module = module.map(lambda x: -x, ptx.Parameter) # MyModule(a=array([-1]), b=array([2]))
```

As shown here, `map` accepts the same `*args` as `filter` and calls `update` at the end if filters are given.

### State Management
Pytreex takes a "direct" approach to state management, i.e., state is updated in-place by the Tree whenever it needs to. For example, this module will calculate the running average of its input:
```python
class Average(ptx.Tree):
    count: ptx.State[jnp.ndarray]
    total: ptx.State[jnp.ndarray]

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
class Dropout(ptx.Tree):
    rng: ptx.Rng[ptx.Initializer, jnp.ndarray]  # Initializer | ndarray

    def __init__(self, rate: float):
        ...
        self.rng = ptx.Initializer(lambda key: key)
        ...

    def __call__(self, x):
        key, self.rng = jax.random.split(self.rng)
        ...
```
Finally `ptx.Optimizer` also performs inplace updates inside the `apply_updates` method, here is a sketch of how it works:
```python
class Optimizer(ptx.TreeObject):
    opt_state: ptx.OptState[Any]
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
If you want to have a static field that contains a non-hashable value like a numpy or jax array, you can use `ptx.Hashable` to wrap around it such that it:

```python
class MyModule(ptx.Tree):
    table: ptx.Hashable[np.ndarray]
    ...

    def __init__(self, table: np.ndarray):
        self.table = ptx.Hashable(table)
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

module.table = ptx.Hashable(np.zeros((10, 10)))

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
import pytreex as ptx

x = np.random.uniform(size=(500, 1))
y = 1.4 * x - 0.3 + np.random.normal(scale=0.1, size=(500, 1))

# pytreex already defines ptx.Linear but we can define our own
class Linear(ptx.Tree):
    w: ptx.Parameter[ptx.Initializer, jnp.ndarray]
    b: ptx.Parameter[jnp.ndarray]

    def __init__(self, din, dout):
        super().__init__()
        self.w = ptx.Initializer(lambda key: jax.random.uniform(key, shape=(din, dout)))
        self.b = jnp.zeros(shape=(dout,))

    def __call__(self, x):
        return jnp.dot(x, self.w) + self.b


model = Linear(1, 1).init(42)
optimizer = ptx.Optimizer(optax.adam(0.01))
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
