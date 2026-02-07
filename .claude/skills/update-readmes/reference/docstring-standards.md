# Docstring Standards

NumPy-style docstrings for all public Python functions and classes.

## Reference

Follow [NumPy docstring guide](https://numpydoc.readthedocs.io/en/latest/format.html).

## Function Docstring

```python
def function_name(
    param1: type1,
    param2: type2,
    param3: type3 | None = None,
) -> ReturnType:
    """Short one-line summary (imperative mood).

    Extended description if needed. Explain WHAT the function does
    and WHY, not HOW (the code shows how).

    Parameters
    ----------
    param1 : type1
        Description of param1.
    param2 : type2
        Description of param2.
    param3 : type3 or None, optional
        Description of param3. Default is None, which means
        <explain default behavior>.

    Returns
    -------
    ReturnType
        Description of what is returned.

    Raises
    ------
    ValueError
        If <condition that triggers the error>.

    See Also
    --------
    related_function : One-line description.

    Notes
    -----
    Any mathematical formulas, algorithm details, or important
    implementation notes.

    Examples
    --------
    >>> result = function_name(arg1, arg2)
    >>> print(result)
    expected_output
    """
```

## Class Docstring

```python
class ClassName:
    """Short one-line summary.

    Extended description.

    Parameters
    ----------
    param1 : type1
        Description (constructor params go here, not in __init__).

    Attributes
    ----------
    attr1 : type
        Description of public attribute.

    See Also
    --------
    RelatedClass : One-line description.
    """

    def __init__(self, param1: type1):
        """Initialize ClassName.

        Only needed if __init__ has complex logic beyond storing params.
        """
```

## Module Docstring

```python
"""Module name - what this module does.

This module provides <functionality> for <purpose>.
Key classes/functions: X, Y, Z.

See Also
--------
related_module : What it does.
"""
```

## Rules

1. **Match signatures exactly**: Use AST to extract actual params, not memory
2. **Imperative mood**: "Compute X" not "Computes X" or "This computes X"
3. **Type annotations**: Use actual Python types, not prose descriptions
4. **Optional params**: Document default behavior, not just default value
5. **No docstrings on private functions**: Skip `_helper()` (except `__init__`)
6. **No docstrings on tests**: Test function names should be self-documenting
7. **See Also for cross-references**: Link related functions in other modules
8. **Raises only if explicit**: Only document exceptions the function raises, not inherited ones

## Priority Tiers

| Tier | Directories | Target Coverage |
|------|-------------|-----------------|
| P0 | `src/data_io/`, `src/viz/`, `src/stats/` | 95%+ |
| P1 | `src/orchestration/`, `src/extraction/` | 90%+ |
| P2 | `src/classification/`, `src/featurization/`, `src/imputation/` | 80%+ |
| P3 | `src/utils/`, `src/log_helpers/`, `src/tools/` | 60%+ |
