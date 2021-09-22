
<!-- ### Opaque static fields -->
When the value of a static field changes `jit` and friends will recompile to reflect posible changes in the computation logic. While this is good quality in many cases, sometimes certain fields are just there for metadata and should not part of the actual computation. For example, `name` in the `Person` example above will probably not affect the logic yet in this example it will force recompilation:


```python
@dataclass
class Person(to.Tree):
    height: float = to.field(node=True)
    name: str = to.field(node=False)

@jax.jit
def do_something(person: Person):
    ...

do_something(Person(height=1.5, name='John')) # compiles
do_something(Person(height=1.5, name='Fred')) # re-compiles 🙁
```
To avoid this, you can mark a field as `opaque`:
```python hl_lines="4"
@dataclass
class Person(to.Tree):
    height: float = to.field(node=True)
    name: str = to.field(node=False, opaque=True)

@jax.jit
def do_something(person: Person):
    ...

do_something(Person(height=1.5, name='John')) # compiles
do_something(Person(height=1.5, name='Fred')) # cached! 🤩
```
`opaque` will "hide" the value content of the field from JAX, changes will only be detected if the type of an opaque field changes or, in case its an array-like type, if its shape or dtype changes.

## opaque_is_equal
If you want to define you on policy on how opaque fields are handled, you can use the `opaque_is_equal: Callable[[to.Opaque, Any], bool]` argument and pass a function that takes in a `to.Opaque(value: Any)` object and the new value of the field and return a boolean indicating whether the new value is considered equal to the opaque value.

```python hl_lines="19"
def same_length(opaque: to.Opaque, other: Any):
    if (
        isinstance(other, to.Opaque) 
        and type(opaque.value) == type(other.value)
    ):
        if isinstance(opaque.value, list):
            return len(opaque.value) == len(other.value)
        else:
            return True
    else:
        return False

@dataclass
class Person(to.Tree):
    height: float = to.field(node=True)
    names: List[str] = to.field(
        node=False, 
        opaque=True, 
        opaque_is_equal=same_length,
    )
```