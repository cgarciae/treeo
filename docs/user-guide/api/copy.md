# Copy

Returns a deep copy of the tree, almost equivalent to:

```python
jax.tree_map(lambda x: x, self)
```

but will try to copy static nodes as well.