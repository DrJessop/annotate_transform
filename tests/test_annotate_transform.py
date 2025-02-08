import keyword
import re
from functools import partial

import jax
import jax.numpy as jnp
import pytest

from annotate_transform import annotate_transform

_keywords = set(keyword.kwlist)


def test_generic_inference_check():
    # Just want to ensure pyright can correctly deduce
    # that the output of the transformation is an array (otherwise would
    # complain that we cannot access shape)
    in_shape = (1, 1, 1)
    in_arr = jnp.ones(in_shape)
    _ = annotate_transform(jnp.sum, "(1, 1, 1) -> ()")(in_arr).shape


def test_rank_must_match():
    in_shape = (1, 1, 1)
    in_arr = jnp.ones(in_shape)

    annotate_transform(jnp.sum, "(1, 1, 1) -> ()")(in_arr)

    with pytest.raises(ValueError, match="Rank of output should be"):
        annotate_transform(jnp.sum, "(1, 1, 1) -> (1,)")(in_arr)

    with pytest.raises(ValueError, match="Rank of input should be"):
        annotate_transform(jnp.sum, "(1, 1) -> (1,)")(in_arr)


def test_shape_must_match():
    in_shape = (1, 1, 1)
    in_arr = jnp.ones(in_shape)
    annotate_transform(jnp.sum, "(b, c, 1) -> (b, 1)")(in_arr, axis=1)
    # Don't explicitly require parentheses around unary operations
    annotate_transform(jnp.sum, "b, c, 1 -> b, 1")(in_arr, axis=1)
    annotate_transform(jnp.sum, "(b, c, 1) -> (d, 1)")(in_arr, axis=1)

    with pytest.raises(ValueError, match="Mismatch between actual input shape and annotated input shape"):
        annotate_transform(jnp.sum, "(b, c, 2) -> (b, 1)")(in_arr, axis=1)

    with pytest.raises(ValueError, match="Mismatch between actual output shape and annotated output shape"):
        annotate_transform(jnp.sum, "(b, c, 1) -> (b, 2)")(in_arr, axis=1)


def test_invalid_binding():
    in_shape = (1, 1, 2)
    in_arr = jnp.ones(in_shape)

    with pytest.raises(ValueError, match="was already bound"):
        annotate_transform(jnp.sum, "(a, a, a) -> ()")(in_arr)

    with pytest.raises(ValueError, match="was already bound"):
        annotate_transform(lambda x: x + 1, "(a, a, b) -> (b, 1, 2)")(in_arr)


def test_container_of_containers():
    def binary_sum(a, b) -> jax.Array:
        return a + b

    a = jnp.ones((1, 1))
    b = jnp.ones((1, 1))
    annotate_transform(binary_sum, "((1, 1), (1, a)) -> (1, 1)")(a, b)

    def return_inputs(*arrs) -> list[jax.Array]:
        out = []
        for arr in arrs:
            assert isinstance(arr, jax.Array)
            out.append(arr)
        return out

    annotate_transform(return_inputs, "((1, 1), (1, 1)) -> ((1, 1), (1, 1))")(a, b)


def test_invalid_annotation_expression():
    with pytest.raises(ValueError, match="Invalid transformation annotation"):
        annotate_transform(lambda x: x, "")(jnp.array([]))

    with pytest.raises(ValueError, match="Invalid transformation annotation"):
        annotate_transform(lambda x: x, "(1, 2) ->")(jnp.array([]))

    with pytest.raises(ValueError, match="Invalid transformation annotation"):
        annotate_transform(lambda x: x, "-> (1, 2)")(jnp.array([]))

    with pytest.raises(ValueError, match="Invalid transformation annotation"):
        annotate_transform(lambda x: x, "-> (1, 2) ->")(jnp.array([]))

    with pytest.raises(ValueError, match="Invalid transformation annotation"):
        annotate_transform(lambda x: x, "((), 2) -> ()")(jnp.array([]))

    with pytest.raises(ValueError, match="Invalid transformation annotation"):
        annotate_transform(lambda x: x, "(((1, 2),), 2) -> ()")(jnp.array([]))

    with pytest.raises(ValueError, match="Invalid transformation annotation"):
        annotate_transform(lambda x: x, "(2, ((1, 2),)) -> ()")(jnp.array([]))

    with pytest.raises(ValueError, match="Invalid transformation annotation"):
        # Only allowed alphanumeric characters
        annotate_transform(lambda x: x, "(a, *) -> ()")(jnp.array([]))

    with pytest.raises(ValueError, match="SyntaxError"):
        # Not wildcard since there is a space after the asterisk
        annotate_transform(lambda x: x, "* a -> * a")(jnp.array([]))

    with pytest.raises(ValueError, match="Invalid transformation annotation"):
        # Cannot use square brackets
        annotate_transform(lambda x: x, "[a, b] -> ()")(jnp.array([]))

    with pytest.raises(ValueError, match="Invalid transformation annotation"):
        # Left side must be a tuple
        annotate_transform(lambda x: x, "b -> ()")(jnp.array([]))

    with pytest.raises(ValueError, match="malformed"):
        # No closing parentheses
        annotate_transform(lambda x: x, "(a, b -> ()")(jnp.array([]))


def test_camelCase_symbolic_dims():
    A = jnp.ones((5, 10))
    B = jnp.ones((10, 20))
    _ = annotate_transform(
        jnp.matmul, "((asteraIsCool1, imbue), (imbue, vastIsCool2)) -> (asteraIsCool1, vastIsCool2)"
    )(A, B)

    # Numbers inside the string should work too
    _ = annotate_transform(
        jnp.matmul, "((astera1IsCool, imbue), (imbue, v2astIsCool)) -> (astera1IsCool, v2astIsCool)"
    )(A, B)


def test_non_array_args():
    A = jnp.zeros((5, 5, 5))
    annotate_transform(jnp.moveaxis, "a,a,a,->a,a,a,")(
        A, 0, 0
    )  # Source and destination are args in the moveaxis API, not kwargs

    # Ensuring I can still manually pass them as kwargs
    annotate_transform(jnp.moveaxis, "a,a,a,->a,a,a,")(A, 0, destination=0)
    annotate_transform(jnp.moveaxis, "a,a,a,->a,a,a,")(A, source=0, destination=0)

    # Non-array args come first
    x = jnp.array([0.0])
    y = jnp.array([1.0])
    annotate_transform(jax.lax.cond, "(a,),(a,) -> a,")(True, lambda x, y: x, lambda x, y: y, x, y)


def test_kwarg_array_arg():
    def fn(a: jax.Array, *, b: jax.Array, _: int, c: jax.Array):
        if a.shape == b.shape:
            return jnp.max(jnp.concat([a, b]))
        return jnp.max(c)

    x = jnp.array([0])
    y = jnp.array([1])
    z = jnp.array([2, 3])
    result = annotate_transform(fn, "(1,),(1,),(2,) -> ()")(x, b=y, _=0, c=z)
    assert result.item() == 1

    # Shapes depend on order they are passed into the function call
    result = annotate_transform(fn, "(1,),(2,),(1,) -> ()")(x, c=z, _=0, b=y)
    assert result.item() == 1

    # For codecov on second branch
    result = annotate_transform(fn, "(1,),(2,),(1,) -> ()")(y, b=z, _=0, c=x)
    assert result.item() == 0


def test_reshape():
    m, k, n = 2, 4, 8
    x = jnp.ones((m, k, n))
    x = annotate_transform(jnp.reshape, "(m, k, n) -> (m * k, n)")(x, (m * k, n))
    assert x.shape == (m * k, n)
    x = annotate_transform(jnp.reshape, "(m * k, n) -> (m, k, n)")(x, (m, k, n))
    assert x.shape == (m, k, n)

    # Want one test to ensure we can parse back to back binary operations
    x = annotate_transform(jnp.reshape, "(m, k, n) -> m * k * n,")(x, (m * k * n,))


def test_concat():
    x = jnp.ones(2)
    y = jnp.ones(5)

    # Would be nice to support pytree annotations so user
    # doesn't have to add these helper functions
    x = annotate_transform(lambda x, y: jnp.concatenate([x, y], axis=0), "(a,),(b,) -> a + b,")(x, y)
    assert x.shape == (7,)


def test_dynamic_slice():
    grid = jnp.ones((2, 2))
    x = annotate_transform(jax.lax.dynamic_slice, "(a, a) -> (a - 1, a - 1)")(grid, (0, 0), (1, 1))
    assert x.shape == (1, 1)


def test_img_resize():
    grid = jnp.ones((5, 5))
    # Notice how '/' is floordiv, not true division. No need to support true division
    # since shapes are integers
    x = annotate_transform(jax.image.resize, "(a, a) -> (a / 2, a / 2)")(grid, (2, 2), method="nearest")
    assert x.shape == (2, 2)


def test_complex_expression():
    inp = jnp.ones((2, 8))

    def fn(_):
        return jnp.ones((4, 4))

    x = annotate_transform(fn, "(a, b + (b / 2) + (b / 2)) -> (a / 2 * 4, b)")(inp)
    assert x.shape == (4, 4)

    # Ensure we can differentiate between wildcards and multiplications
    inp = jnp.ones((2, 8))

    def fn2(_):
        return jnp.ones((2, 80))

    x = annotate_transform(fn2, "(a, b) -> (a, b * (2 + b))")(inp)
    assert x.shape == (2, 80)

    with pytest.raises(ValueError, match=re.escape("Could not evaluate b * (2 + b + 1) to 88. Got 80.")):
        annotate_transform(fn2, "(a, b) -> (a, b * (2 + b + 1))")(inp)


def test_keyword_in_annotation():
    inp = jnp.ones(2)
    for kw in _keywords:
        x = annotate_transform(lambda x: x, f"{kw}, -> {kw},")(inp)
        assert x.shape == (2,)

    # Ensure error message contains keyword, not keyword prefixed with underscore which
    # we use for internally replacing keywords
    inp2 = jnp.ones(3)
    for kw in _keywords:
        with pytest.raises(ValueError, match=f"{kw} was already bound"):
            annotate_transform(lambda x: inp2, f"{kw}, -> {kw},")(inp)


def test_wildcards():
    batch_size = 2
    sequence_length = 3
    pytree = {
        "a": jnp.ones((batch_size, sequence_length, 1, 1)),
        "b": jnp.ones((batch_size, sequence_length, 2, 2, 2)),
    }

    @partial(annotate_transform, annotation="batch, seq, *feat -> *feat,")
    def transform(arr: jax.Array) -> jax.Array:
        return jnp.sum(arr, axis=(0, 1))

    transformed_pytree = jax.tree.map(transform, pytree)
    for key, value in transformed_pytree.items():
        assert value.shape == pytree[key].shape[2:]

    pytree = {
        "a": jnp.ones((batch_size, sequence_length, 1, 1, 5, 6, 8, 8, 1, 1)),
        "b": jnp.ones((batch_size, sequence_length, 2, 2, 2, 5, 6, 8, 8, 8, 8, 2, 2, 2)),
    }

    @partial(annotate_transform, annotation="(batch, seq, *a), (batch, seq, *a,) -> *a,")
    def complex_transform(arr: jax.Array, arr2: jax.Array) -> jax.Array:
        return jnp.sum(arr, axis=(0, 1))

    transformed_pytree = jax.tree.map(complex_transform, pytree, pytree)

    @partial(annotate_transform, annotation="1, *a, *b -> 2, *a, *b")
    def invalid_transform(arr: jax.Array) -> jax.Array:
        return arr

    with pytest.raises(ValueError, match="Cannot have multiple wildcards in a single shape"):
        jax.tree.map(invalid_transform, {"a": jnp.ones((1, 1, 1))})

    @partial(annotate_transform, annotation="1, 1, 1, 1, *a -> 1, 1, 1")
    def invalid_transform_input_rank_error(arr: jax.Array) -> jax.Array:
        return arr

    with pytest.raises(
        ValueError, match="Rank of concrete dimensions is larger than the actual shape, so cannot match wildcard"
    ):
        jax.tree.map(invalid_transform_input_rank_error, {"a": jnp.ones((1, 1, 1))})

    @partial(annotate_transform, annotation="1, 1, 1 -> 1, 1, 1, 1, *a")
    def invalid_transform_output_rank_error(arr: jax.Array) -> jax.Array:
        return arr

    with pytest.raises(
        ValueError, match="Rank of concrete dimensions is larger than the actual shape, so cannot match wildcard"
    ):
        jax.tree.map(invalid_transform_output_rank_error, {"a": jnp.ones((1, 1, 1))})

    @partial(annotate_transform, annotation="1, a, *a -> 1, a, *a")
    def invalid_transform_symbolic_dim_and_wildcard_with_same_name(arr: jax.Array) -> jax.Array:
        return arr

    with pytest.raises(ValueError, match="Cannot have a variable being used as a wildcard and as a concrete dimension"):
        invalid_transform_symbolic_dim_and_wildcard_with_same_name(jnp.ones(1))

    @partial(annotate_transform, annotation="1, *1 -> 1, *1")
    def invalid_transform_bad_wildcard(arr: jax.Array) -> jax.Array:
        return arr

    with pytest.raises(ValueError, match="must be a legal Python variable name"):
        invalid_transform_bad_wildcard(jnp.ones(1))
