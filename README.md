# Annotate Transform

A Python library for annotating and validating shape transformations in JAX arrays.

## Overview

`annotate_transform` provides a decorator that allows you to specify expected input and output shapes for JAX array transformations. This helps catch shape-related bugs early and makes code more self-documenting. It is recommended to only use this in jitted functions
so that it only runs when tracing.

## Installation
From pypi
```bash
pip install annotate-transform
```

From source
```bash
uv sync
```

## Running tests
```bash
uv run tests/test_annotate_transform.py
```

## Usage 
```python
def annotate_transform(
    transform: Callable[_inputs_type, _outputs_type], annotation: str
) -> Callable[_inputs_type, _outputs_type]:
    """Annotates and checks transformations to jax.Arrays when the returned
    function is invoked.
    If annotation does not match the actual transform, raises ValueError.

    Example:
        >>> in_shape = (5, 3, 24, 24)
        >>> a = jnp.ones(in_shape)
        >>> b = annotate_transform(jnp.sum, "(b, c, h, w) -> (b, h, w)")(a, axis=1)
        >>> c = annotate_transform(jnp.sum, "(5, 3, 24, 24) -> (b, h, 24)")(a, axis=1)
        >>> assert (b == c).all()

    Matmul example:
        >>> A = jnp.ones((5, 10))
        >>> B = jnp.ones((10, 5))
        >>> C = annotate_transform(jnp.matmul, "((a, b), (b, c)) -> (a, c)")(A, B)

        >>> # The following will work as well, although it's less readable than the above
        >>> # since we know that for matmuls, 'd' will always be 'b'
        >>> works = annotate_transform(jnp.matmul, "((a, b), (d, c)) -> (a, c)")(A, B)

        >>> # The following will error out, since we already bound 'a' to 5, and we are trying
        >>> # to reuse it in place of 10
        >>> error = annotate_transform(jnp.matmul, "((a, a), (b, c)) -> (a, c)")(A, B)

    Symbolic dim convention:
        >>> A = jnp.ones((5, 10))
        >>> B = jnp.ones((10, 5))
        >>> # Symbolic dims can be multicharacter
        >>> C = annotate_transform(jnp.matmul, "((bananaMan, b), (b, potatoMan)) -> (bananaMan, potatoMan)")(A, B)
        >>> # Snake case will work as well
        >>> C = annotate_transform(jnp.matmul, "((b_man, b), (b, p_man)) -> (b_man, p_man)")(A, B)

    Functions with keyword arguments:
        >>> # The order that arguments and keyword arguments are provided is how this function validates
        >>> # the shape transformation. For example, consider the following function
        >>> def fn(a: jax.Array, *, b: jax.Array, _: int, c: jax.Array):
        ...   if a.shape == b.shape:
        ...     return jnp.max(jnp.concat([a, b]))
        ...   return c
        ...
        >>> # This function has 1 positional argument and 3 keyword arguments, one of those keyword
        >>> # arguments being an integer instead of a jax.Array.
        >>> # Now consider these invocations
        >>> x = jnp.array([0])
        >>> y = jnp.array([1])
        >>> z = jnp.array([1, 2])
        >>> # To validate the inputs, this function just iterates over all positional and keyword
        >>> # arguments and collects all jax.Array types.
        >>> # Notice how we pass in 3 shapes, even though there are 4 arguments passed in total.
        >>> result = annotate_transform(fn, "(1,),(1,),(2,) -> ()")(x, b=y, _=0, c=z)
        >>> # If we change the order of the keyword arguments, we also need to change the shape
        >>> # annotation to reflect that new ordering
        >>> result = annotate_transform(fn, "(1,),(2,),(1,) -> ()")(x, c=z, _=0, b=y)
        >>> # For easier readability, it's suggested to place all your non-array keyword arguments
        >>> # at the end, like so
        >>> result = annotate_transform(fn, "(1,),(2,),(1,) -> ()")(x, c=z, b=y, _=0)

    Mathematical expressions in shape annotation:
        >>> # Certain shape transformations are functions of other dimensions. For example,
        >>> # reshape must preserve the hypervolume of the input array. Thus, we support
        >>> # mathematical expressions in the shape annotation.
        >>> result = annotate_transform(jnp.reshape, "(b, h, w) -> b * h * w,")(jnp.ones((5, 3, 24)), -1)
        >>> # One subtlety is that you must ensure that dims involved in a mathematical expression
        >>> # are bound at some point in the shape annotation. Otherwise, this will error out.
        >>> # Here is a case that works, in which we check that the expression is bound later on.
        >>> result = annotate_transform(jnp.reshape, "(b * h * w), -> b, h, w")(jnp.ones((5 * 3 * 24)), (5, 3, 24))
        >>> # However, this will error out, since we never bound 'b'.
        >>> result = annotate_transform(jnp.reshape, "(b * h * w), -> c, h, w")(jnp.ones((5 * 3 * 24)), (5, 3, 24))
        >>> # So far, the only mathematical expressions that are supported are multiplication, division, addition,
        >>> # and subtraction. Note, that for division, we use the symbol '/' instead of the usual '//', but we will
        >>> # perform floor division under the hood.

    Wildcard support in shape annotation:
        >>> batch_size = 2
        >>> sequence_length = 3
        >>> pytree = {
        ...     "a": jnp.ones((batch_size, sequence_length, 1, 1)),
        ...     "b": jnp.ones((batch_size, sequence_length, 2, 2, 2)),
        ... }
        >>> # Just like with symbolic dimensions, wildcards are bound to the same shape
        >>> # for the duration of the transform annotation check.
        >>> @partial(annotate_transform, annotation="batch, seq, *feat -> *feat,")
        ... def transform(arr: jax.Array) -> jax.Array:
        ...     return jnp.sum(arr, axis=(0, 1))
        >>> transformed_pytree = jax.tree.map(transform, pytree)
        >>> # Note that we only support up to one wildcard per shape, and there
        >>> # cannot be a space after the asterisk! Additionally, we cannot have
        >>> # wildcard variables with the same name as a concrete dimension.
        >>> # Here are examples that will fail:
        >>> @partial(annotate_transform, annotation="batch, seq, * feat -> * feat,")
        ... def transform_with_space_after_wildcard(arr: jax.Array) -> jax.Array:
        ...     # This will error out since there is a space after the asterisk
        ...     return jnp.sum(arr, axis=(0, 1))
        >>> @partial(annotate_transform, annotation="a, b, *b -> *b,")
        ... def transform_with_wildcard_same_name_as_concrete_dim(arr: jax.Array) -> jax.Array:
        ...     # This will error out since we cannot have a wildcard variable with the same name
        ...     # as a concrete dimension
        ...     return jnp.sum(arr, axis=(0, 1))
        >>> @partial(annotate_transform, annotation="((a, *b), (*c,)) -> ((a, *b), (*c,))")
        ... def transform_with_one_wildcard_per_shape(arr: jax.Array, arr2: jax.Array) -> jax.Array:
        ...     # This is fine because we have only one wildcard per shape
        ...     return arr, arr2
        >>> @partial(annotate_transform, annotation="((a, *b), (*c, *d)) -> ((a, *b), (*c, *d))")
        ... def transform_with_multiple_wildcards_in_same_shape(arr: jax.Array, arr2: jax.Array) -> jax.Array:
        ...     # This will error out since we have multiple wildcards in the same shape
        ...     return arr, arr2
        >>> # Finally, we cannot use wildcards on concrete dimensions. Here is an example that will fail:
        >>> @partial(annotate_transform, annotation="(a, *1 -> a, *1)")
        ... def transform_with_wildcard_on_concrete_dim(arr: jax.Array) -> jax.Array:
        ...     # This will error out since we cannot use wildcards on concrete dimensions
        ...     return arr

    Note: This is expensive to do at runtime, so if using this function, make sure to jit the caller function.

    """
    return partial(_transform_and_check, transform, annotation)
```
