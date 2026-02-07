"""
Custom exceptions for statistics module.

Provides a hierarchy of exceptions for clear error handling
and informative error messages.

Cross-references:
- planning/statistics-implementation.md (Section 1.3)
"""


class StatsError(Exception):
    """
    Base exception for all statistics module errors.

    All custom exceptions in this module inherit from StatsError,
    allowing callers to catch all statistics-related errors with
    a single except clause if desired.

    Examples
    --------
    >>> try:
    ...     # some statistics computation
    ...     pass
    ... except StatsError as e:
    ...     print(f"Statistics error: {e}")
    """

    pass


class InsufficientDataError(StatsError):
    """
    Raised when sample size is too small for the requested computation.

    Attributes
    ----------
    required : int
        Minimum required sample size
    actual : int
        Actual sample size provided
    context : str
        Additional context about the computation

    Examples
    --------
    >>> raise InsufficientDataError(required=30, actual=5, context="bootstrap CI")
    InsufficientDataError: Insufficient data: required 30, got 5. bootstrap CI
    """

    def __init__(self, required: int, actual: int, context: str = ""):
        """
        Initialize InsufficientDataError with sample size information.

        Parameters
        ----------
        required : int
            Minimum required sample size for the computation.
        actual : int
            Actual sample size provided.
        context : str, optional
            Additional context about the computation (e.g., "bootstrap CI").
        """
        self.required = required
        self.actual = actual
        self.context = context
        message = f"Insufficient data: required {required}, got {actual}"
        if context:
            message += f". {context}"
        super().__init__(message)


class SingleClassError(StatsError):
    """
    Raised when only one class is present in binary classification data.

    This commonly occurs with small samples or extreme class imbalance.

    Attributes
    ----------
    class_counts : dict
        Dictionary mapping class labels to their counts

    Examples
    --------
    >>> raise SingleClassError(class_counts={0: 50, 1: 0})
    SingleClassError: Single class present: {0: 50, 1: 0}. Need both classes for binary classification.
    """

    def __init__(self, class_counts: dict):
        """
        Initialize SingleClassError with class distribution information.

        Parameters
        ----------
        class_counts : dict
            Dictionary mapping class labels to their counts.
            For example, {0: 50, 1: 0} indicates all samples belong to class 0.
        """
        self.class_counts = class_counts
        super().__init__(
            f"Single class present: {class_counts}. "
            "Need both classes for binary classification."
        )


class ConvergenceError(StatsError):
    """
    Raised when an optimization or fitting procedure fails to converge.

    Attributes
    ----------
    method : str
        Name of the method that failed
    iterations : int
        Number of iterations attempted

    Examples
    --------
    >>> raise ConvergenceError(method="logistic regression", iterations=1000)
    ConvergenceError: logistic regression failed to converge after 1000 iterations.
    """

    def __init__(self, method: str, iterations: int):
        """
        Initialize ConvergenceError with method and iteration information.

        Parameters
        ----------
        method : str
            Name of the optimization or fitting method that failed to converge.
        iterations : int
            Number of iterations attempted before failure.
        """
        self.method = method
        self.iterations = iterations
        super().__init__(f"{method} failed to converge after {iterations} iterations.")


class CheckpointError(StatsError):
    """
    Raised when checkpoint loading or saving fails.

    Attributes
    ----------
    operation : str
        'load' or 'save'
    path : str
        Checkpoint file path
    reason : str
        Reason for failure
    """

    def __init__(self, operation: str, path: str, reason: str = ""):
        """
        Initialize CheckpointError with operation details.

        Parameters
        ----------
        operation : str
            The operation that failed, typically 'load' or 'save'.
        path : str
            File path of the checkpoint that caused the error.
        reason : str, optional
            Additional reason for the failure (e.g., "file not found",
            "corrupted data").
        """
        self.operation = operation
        self.path = path
        self.reason = reason
        message = f"Checkpoint {operation} failed for '{path}'"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class ValidationError(StatsError):
    """
    Raised when input validation fails.

    Attributes
    ----------
    parameter : str
        Name of the invalid parameter
    expected : str
        Description of expected value/format
    actual : str
        Description of actual value received
    """

    def __init__(self, parameter: str, expected: str, actual: str):
        """
        Initialize ValidationError with parameter validation details.

        Parameters
        ----------
        parameter : str
            Name of the parameter that failed validation.
        expected : str
            Description of the expected value or format
            (e.g., "positive integer", "array of shape (n, 2)").
        actual : str
            Description of the actual value received
            (e.g., "-5", "array of shape (n, 3)").
        """
        self.parameter = parameter
        self.expected = expected
        self.actual = actual
        super().__init__(f"Invalid '{parameter}': expected {expected}, got {actual}")


class DegenerateCaseError(StatsError):
    """
    Raised when a computation encounters a degenerate case.

    Examples include zero variance, perfect collinearity, or
    other edge cases that make the computation undefined.

    Attributes
    ----------
    computation : str
        Name of the computation
    reason : str
        Description of the degenerate case
    """

    def __init__(self, computation: str, reason: str):
        """
        Initialize DegenerateCaseError with computation details.

        Parameters
        ----------
        computation : str
            Name of the computation that encountered the degenerate case.
        reason : str
            Description of why the case is degenerate
            (e.g., "zero variance in data", "perfect collinearity",
            "all predictions identical").
        """
        self.computation = computation
        self.reason = reason
        super().__init__(f"Degenerate case in {computation}: {reason}")
