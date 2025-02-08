import ast
import keyword
import operator as op
import re
from ast import NodeVisitor
from collections.abc import Callable, Sequence
from functools import partial
from typing import ParamSpec, TypeVar

import jax

_keywords = set(keyword.kwlist)


class _Wildcard:
    def __init__(self, name: str, shape_index: int, axis_index: int):
        self.name = name
        self.shape_index = shape_index
        self.axis_index = axis_index

    def __repr__(self):
        return f"*{self.name}"


_axis_type = int | str | Callable | _Wildcard
_shape_type = tuple[_axis_type, ...]
_shapes_type = tuple[_shape_type]
_inputs_type = ParamSpec("_inputs_type")
_outputs_type = TypeVar("_outputs_type", bound=jax.Array | Sequence[jax.Array])

_operators = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.floordiv,
}


class _EvalVisitor(NodeVisitor):
    def __init__(self, **kwargs):
        self._namespace = kwargs

    def visit_Name(self, node):
        return self._namespace[node.id]

    def visit_Constant(self, node: ast.Constant):
        return node.value

    def visit_UnaryOp(self, node: ast.UnaryOp):
        raise NotImplementedError("UnaryOps not supported")

    def visit_BinOp(self, node: ast.BinOp):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        operator_type = type(node.op)
        if operator_type not in _operators:
            raise ValueError(f"Unsupported binary operator: {operator_type}")
        return _operators[operator_type](lhs, rhs)

    def generic_visit(self, node):
        raise ValueError("malformed node or string: " + repr(node))


def _parse_annotation(annotation: str) -> tuple[_shapes_type, _shapes_type]:
    annotation = annotation.strip()
    try:
        input_shapes, output_shapes = annotation.split("->")
    except ValueError as err:
        raise ValueError("Invalid transformation annotation: must be '{{input_shapes}} -> {{output_shapes}}'") from err

    input_shapes = input_shapes.strip()
    output_shapes = output_shapes.strip()
    if not len(input_shapes) or not len(output_shapes):
        raise ValueError("Invalid transformation annotation: must have input and output shapes")

    for char in input_shapes + output_shapes:
        if not char.isalnum() and char not in (",", "(", ")", " ", "*", "-", "/", "+", "_"):
            raise ValueError(f"Invalid transformation annotation: invalid character: {char}")

    def _validate_and_standardize_shapes(node: ast.Tuple) -> ast.Tuple:
        # We need to make sure that the node is either a flat tuple or a tuple of flat tuples
        at_least_one_nested_tuple = False
        at_least_one_non_tuple = False
        for elt in node.elts:
            if isinstance(elt, ast.Tuple):
                at_least_one_nested_tuple = True
                for subelt in elt.elts:
                    if isinstance(subelt, ast.Tuple):
                        raise ValueError("Invalid transformation annotation. Nested tuples are not allowed.")
            else:
                at_least_one_non_tuple = True

        if at_least_one_nested_tuple and at_least_one_non_tuple:
            raise ValueError("Invalid transformation annotation. Cannot mix tuples and non-tuples in shape annotation.")

        if at_least_one_non_tuple:
            node = ast.Tuple([node])

        return node

    def _replace_wildcards(shape: str) -> tuple[str, set[str]]:
        # Replace wildcards (*) with unique variable names
        # Returns the modified shape string and a mapping from original wildcards to variable names
        wildcards = set()
        dimensions = [t for t in re.findall(r"([(,)]|[^[(,)\s]+)", shape) if t.strip()]
        new_dims = []
        for dim in dimensions:
            if dim.startswith("*") and len(dim) > 1:
                dim = dim[1:]
                # Check that wildcard variable name is valid
                if not re.match(r"^[a-zA-Z_]\w*$", dim):
                    raise ValueError(
                        "Invalid transformation annotation: wildcard variable name must be a legal Python variable name"
                    )
                wildcards.add(dim)

            new_dims.append(dim)

        for dim in dimensions:
            if dim in wildcards and not dim.startswith("*"):
                raise ValueError("Cannot have a variable being used as a wildcard and as a concrete dimension")

        return "".join(new_dims), wildcards

    def _parse_shape_string(shape: str, ret_to_wildcard: dict[str, _Wildcard]) -> _shapes_type:
        shape, wildcards = _replace_wildcards(shape)
        node = ast.parse(shape, mode="eval").body
        if not isinstance(node, ast.Tuple):
            raise ValueError("Invalid transformation annotation. Node must be a tuple.")

        if not node.elts:
            return ((),)

        # We need to make sure that the node is either a flat tuple or a tuple of flat tuples
        node = _validate_and_standardize_shapes(node)

        processed = []

        def _construct_lambda(variables: list[str], bin_op: ast.BinOp, values: list[int]):
            # Convert the bin_op into a lambda which searches and replaces all named
            # variables with their corresponding values
            kv = {k: v for k, v in zip(variables, values, strict=True)}

            # Providing the unparsed expression is useful for debugging
            return _EvalVisitor(**kv).visit(bin_op), ast.unparse(bin_op)

        def _process_bin_op(bin_op: ast.BinOp):
            # Convert the binary operation into a lambda function
            # which takes all named variables as input
            variables = []
            for node in ast.walk(bin_op):
                if isinstance(node, ast.Name):
                    variables.append(node.id)
            return partial(_construct_lambda, variables, bin_op)

        for shape_index, elt in enumerate(node.elts):
            assert isinstance(elt, ast.Tuple | ast.List)
            tuple_content = []

            has_wildcard = False
            for axis_index, subelt in enumerate(elt.elts):
                if isinstance(subelt, ast.Constant):
                    ret = subelt.value
                elif isinstance(subelt, ast.Name):
                    ret = subelt.id
                    if ret in wildcards:
                        if has_wildcard:
                            raise ValueError("Cannot have multiple wildcards in a single shape")
                        has_wildcard = True
                        ret = _Wildcard(ret, shape_index, axis_index)
                        if ret.name not in ret_to_wildcard:
                            ret_to_wildcard[ret.name] = ret

                elif isinstance(subelt, ast.BinOp):
                    ret = _process_bin_op(subelt)
                else:
                    raise ValueError(f"Invalid shape annotation. Illegal operation: {subelt}")
                tuple_content.append(ret)
            processed.append(tuple(tuple_content))
        return tuple(processed)

    ret_to_wildcard: dict[str, _Wildcard] = {}
    try:
        ins = _parse_shape_string(input_shapes, ret_to_wildcard)
        outs = _parse_shape_string(output_shapes, ret_to_wildcard)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError, AssertionError) as err:
        raise ValueError(
            f"Invalid transformation annotation due to malformed annotation: {type(err).__name__}({err.args[0]})"
        ) from err

    return ins, outs


def _replace_keywords(annotation: str) -> tuple[str, dict[str, str]]:
    replaced_keywords = {}
    for kw in _keywords:
        pattern = rf"\b{kw}\b(?![a-zA-Z0-9])"
        if re.search(pattern, annotation):
            new_kw = f"_{kw}"
            annotation = re.sub(pattern, new_kw, annotation)
            replaced_keywords[new_kw] = kw

    return annotation, replaced_keywords


def _transform_and_check(
    transform: Callable[_inputs_type, _outputs_type],
    annotation: str,
    *args: _inputs_type.args,
    **kwargs: _inputs_type.kwargs,
) -> _outputs_type:
    annotation, replaced_keywords = _replace_keywords(annotation)
    expected_in_shapes, expected_out_shapes = _parse_annotation(annotation)

    arrays = []
    for arg in args:
        if isinstance(arg, jax.Array):
            arrays.append(arg)

    for kwarg in kwargs.values():
        if isinstance(kwarg, jax.Array):
            arrays.append(kwarg)

    trans_out = transform(*args, **kwargs)
    trans_out_container = trans_out
    if isinstance(trans_out, jax.Array):
        trans_out_container = (trans_out,)

    trans_out_shapes: _shapes_type = tuple(t.shape for t in trans_out_container)  # type: ignore
    trans_in_shapes = tuple(t.shape for t in arrays)

    def attempt_to_match_candidate(expected_in_shapes, expected_out_shapes):
        bound_dims = dict()
        expression_and_return_value = dict()
        for recv_shape, exp_shape in zip(trans_in_shapes, expected_in_shapes, strict=True):
            for recv_dim, exp_dim in zip(recv_shape, exp_shape, strict=True):
                if isinstance(exp_dim, int):
                    if recv_dim != exp_dim:
                        raise ValueError(
                            "Mismatch between actual input shape and annotated "
                            f"input shape. Actual: {recv_shape}, annotated {exp_shape}"
                        )
                elif isinstance(exp_dim, str):
                    if exp_dim not in bound_dims:
                        bound_dims[exp_dim] = recv_dim
                    elif bound_dims[exp_dim] != recv_dim:
                        _exp_dim = replaced_keywords.get(exp_dim, exp_dim)
                        raise ValueError(
                            f"{_exp_dim} was already bound to {bound_dims[exp_dim]}, trying to bind to {recv_dim}"
                        )
                elif isinstance(exp_dim, Callable):
                    expression_and_return_value[exp_dim] = recv_dim

        for recv_shape, exp_shape in zip(trans_out_shapes, expected_out_shapes, strict=True):
            for recv_dim, exp_dim in zip(recv_shape, exp_shape, strict=True):
                if isinstance(exp_dim, int):
                    if recv_dim != exp_dim:
                        raise ValueError(
                            "Mismatch between actual output shape and annotated "
                            f"output shape. Actual: {recv_shape}, annotated {exp_shape}"
                        )
                elif isinstance(exp_dim, str):
                    if exp_dim not in bound_dims:
                        bound_dims[exp_dim] = recv_dim
                    elif bound_dims[exp_dim] != recv_dim:
                        _exp_dim = replaced_keywords.get(exp_dim, exp_dim)
                        raise ValueError(
                            f"{_exp_dim} was already bound to {bound_dims[exp_dim]}, trying to bind to {recv_dim}"
                        )
                elif isinstance(exp_dim, Callable):
                    expression_and_return_value[exp_dim] = recv_dim

        for expression, return_value in expression_and_return_value.items():
            dims = expression.args[0]
            for dim in dims:
                if dim not in bound_dims:
                    raise ValueError(f"Could not evaluate {expression} because {dim} was not bound")
            expected_return_value, unparsed_expression = expression([bound_dims[dim] for dim in dims])
            if return_value != expected_return_value:
                raise ValueError(
                    f"Could not evaluate {unparsed_expression} to {expected_return_value}. Got {return_value}."
                )

    def _maybe_replace_wildcard(expected_shapes: _shapes_type, actual_shapes: _shapes_type) -> _shapes_type:
        new_expected_shapes = []
        for expected_shape in expected_shapes:
            new_expected_shape = []
            for dim in expected_shape:
                if isinstance(dim, _Wildcard):
                    actual_shape_length = len(actual_shapes[dim.shape_index])
                    expected_shape_concrete_length = len(expected_shape) - 1
                    wildcard_length = actual_shape_length - expected_shape_concrete_length
                    if wildcard_length < 0:
                        raise ValueError(
                            "Rank of concrete dimensions is larger than the actual shape, so cannot match wildcard"
                        )
                    for i in range(dim.axis_index, dim.axis_index + wildcard_length):
                        new_expected_shape.append(actual_shapes[dim.shape_index][i])
                else:
                    new_expected_shape.append(dim)
            new_expected_shapes.append(tuple(new_expected_shape))
        return tuple(new_expected_shapes)

    expected_in_shapes = _maybe_replace_wildcard(expected_in_shapes, trans_in_shapes)
    expected_out_shapes = _maybe_replace_wildcard(expected_out_shapes, trans_out_shapes)

    for recv_shape, exp_shape in zip(trans_in_shapes, expected_in_shapes, strict=True):
        if len(recv_shape) != len(exp_shape):
            raise ValueError(f"Rank of input should be {len(exp_shape)}, got {len(recv_shape)}")

    for recv_shape, exp_shape in zip(trans_out_shapes, expected_out_shapes, strict=True):
        if len(recv_shape) != len(exp_shape):
            raise ValueError(f"Rank of output should be {len(exp_shape)}, got {len(recv_shape)}")

    attempt_to_match_candidate(expected_in_shapes, expected_out_shapes)

    return trans_out


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
