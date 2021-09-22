
<!-- ### Node policy -->
If a field is **not** marked with `to.field` the following policy will be applied when determining whether a field is a node or not:

* If the field is _annotated_ with a `Tree` subtype or a generic containing a `Tree` subtype e.g. `List[to.Tree]`, the field is considered a **node**.
* If the runtime value of the field is a `to.Tree` instance, the field is considered a **node** and the `Tree`'s metadata will be updated to reflect this.
* If none of the above apply, the field is considered a **static** field.

```python
import treeo as to

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
