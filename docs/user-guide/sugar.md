<!-- ### Sugar 🍬 -->
For pedagogical reason, `field` has been used throught the documentation to reinforce the concepts, however, Treeo has a couple of shortcuts to make it easier and more understandable to define Trees.

| normal                            | shortcut        |
| --------------------------------- | --------------- |
| `to.field(node=True)`             | `to.node()`     |
| `to.field(node=False)`            | `to.static()`   |
| `to.field(node=True, kind=Kind)`  | `Kind.node()`   |
| `to.field(node=False, kind=Kind)` | `Kind.static()` |

Based on this, you can take the following code

```python
class Parameter:
    pass

class Child(to.Tree):
    a: float = to.field(node=True)
    b: str = to.field(node=False)
    b: float = to.field(node=True, kind=Parameter)
    d: float = to.field(node=False, kind=Parameter)

class Parent(to.Tree):
    child1: Child = to.field(node=True)
    child2: Child = to.field(node=False)
    rest: List[Child] = to.field(node=True)
```

and simplify it to:

```python
class Parameter(to.KindMixin): pass

class Child(to.Tree):
    a: float = to.node()
    b: str # = to.static(), inferred
    b: float = Parameter.node()
    d: float = Parameter.static()

class Parent(to.Tree):
    child1: Child # = to.node(), inferred
    child2: Child = to.static()
    rest: List[Child] # = to.node(), inferred
```

The `to.KindMixin` provides the `.field()`, `.node()`, and `.static()` methods to subtypes, in this case the `Parameter` class.
