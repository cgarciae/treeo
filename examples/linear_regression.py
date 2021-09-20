from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import treeo as to


class Linear(to.Tree):
    w: jnp.ndarray = to.node()
    b: jnp.ndarray = to.node()

    def __init__(self, din, dout, key: Any):
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
