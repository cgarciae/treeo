<!-- ### Sugar 🍬 -->
For pedagogical reason, `field` has been used throught the documentation to reinforce the concepts, however, Treeo has a couple of shortcuts to make it easier and more understandable to define Trees.

| normal                                  | shortcut              |
| --------------------------------------- | --------------------- |
| `to.field(node=True)`                   | `to.node()`           |
| `to.field(node=False)`                  | `to.static()`         |
| `to.field(node=True, kind=TreeOrKind)`  | `TreeOrKind.node()`   |
| `to.field(node=False, kind=TreeOrKind)` | `TreeOrKind.static()` |

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
    child1: Child = to.field(node=True, kind=Child)
    child2: Child = to.field(node=False, kind=Child)
    rest: List[Child] = to.field(node=True, kind=Child)
```
and simplify it to:
```python
class Parameter(to.KindMixin):
    pass

class Child(to.Tree):
    a: float = to.node()
    b: str # = to.static(), inferred
    b: float = Parameter.node()
    d: float = Parameter.static()

class Parent(to.Tree):
    child1: Child # = Child.node(), inferred
    child2: Child = Child.static()
    rest: List[Child] # = Child.node(), inferred
```
As you see we use the `to.KindMixin` to add some methods to the `Parameter` class. Also, since the `Tree` already inherits from `to.KindMixin`, we can use the `.node()` and `.static()` methods on the `Tree` subclass.
