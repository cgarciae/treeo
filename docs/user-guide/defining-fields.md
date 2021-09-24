
<!-- ### Tree fields -->

Tree fields are divided into two categories:

* `node` fields: they are considered as part of the pytree, JAX functions such as `tree_map` will operate over them.
* `static` fields: they are part of the `PyTreeDef`, JAX functions will not operate over them, but JAX is still aware of them, e.g. JAX will recompile jitted functions is case these fields change.


```python hl_lines="5-6"
import treeo as to

@dataclass
class Person(to.Tree):
    height: float = to.field(node=True)
    name: str = to.field(node=False)

person = Person(height=1.5, name='John')

tree_map(lambda x: x + 1, person) # Person(height=2.5, name='John')
```
Since `field` is just a wrapper over `dataclasses.field` that adds the `node` and `kind` arguments you can use all `dataclass` features. However, dataclasses are orthogonal to Treeo, this means that you can naturally use non-dataclass classes:

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

