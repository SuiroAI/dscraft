"""Picklable module-level callables used by test_seatbelt.py's run_callable tests.

These must live in their own module (not inline in test_seatbelt.py) so
that :mod:`pickle` can reference them by ``(module, qualname)`` and the
sandboxed subprocess -- which runs a fresh Python interpreter, not the
pytest process -- can re-import this module (via ``PYTHONPATH`` pointed at
this directory) to unpickle them. This file is not itself a test module
(no ``test_`` prefix), so pytest will not try to collect it.
"""

from __future__ import annotations


def compute_answer() -> int:
    """Return 42, a trivial picklable callable for run_callable() success tests."""
    return 6 * 7


def raise_value_error() -> None:
    """Raise ValueError, a trivial picklable callable for run_callable() failure tests."""
    raise ValueError("boom from sandboxed callable")


def add_numbers(a: int, b: int = 0) -> int:
    """Return ``a + b``, a module-level function used to exercise
    run_callable()'s functools.partial + JSON-serializable-args calling
    convention (Finding 2 fix)."""
    return a + b


def describe_payload(data: dict, tags: list, note: str, count: int, ratio: float, flag: bool, empty: object) -> str:
    """Return a deterministic repr() of every argument, used to confirm
    end-to-end that a legitimate mix of JSON-native argument types (dict
    with string keys, list, str, int, float, bool, None) still round-trips
    correctly through run_callable()'s stricter args/kwargs validation
    (Finding 3 fix)."""
    return repr((data, tags, note, count, ratio, flag, empty))
