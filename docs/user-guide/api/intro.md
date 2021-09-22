Throught these examples for the functional API we will use the following defintions:
```python
import treeo as to

class Parameter:
    pass

class BatchStat:
    pass

@dataclass
class MyTree(to.Tree):
    a: jnp.ndarray = to.field(node=True, kind=Parameter)
    b: jnp.ndarray = to.field(node=True, kind=BatchStat)
```