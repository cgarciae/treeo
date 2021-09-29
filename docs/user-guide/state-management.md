

<!-- ### State Management -->
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


## What is the catch?
<!-- TODO: Add a list of rules to follow around jitted functions -->
State management is one of the most challenging things in JAX, but with the help of Treeo it seems effortless, what is the catch? As always there is a trade-off to consider: Treeo's approach requires to consider how to propagate state changes properly while taking into account the fact that Pytree operations create new objects, that is, since reference do not persist across calls through these functions changes might be lost. 

A standard solution to this problem is: **always output the Tree to update state**. For example, a typical gradient function in a Deep Learning application that contains a stateful Tree would look like this:

```python
@partial(jax.value_and_grad, has_aux=True)
def grad_fn(params, model, x, y):
    model = to.merge(model, params)

    y_pred = model(x) # state is updated
    loss = jnp.mean((y_pred - y) ** 2)

    return loss, model # return model to propagate state changes

params = to.filter(model, Parameter)
(loss, model), grads = grad_fn(params, model, x, y)
...
```
Here `model` is returned along with the loss through `value_and_grad` to update `model` on the outside thus persisting any changes to the state performed on the inside.