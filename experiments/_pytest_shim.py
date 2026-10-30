"""A very small stand-in for the parts of pytest this repository's tests use.

pytest cannot be installed in this environment (no package-index access), so
without something standing in its place every test module that does
``import pytest`` fails at import and the suite reports almost nothing. This
shim supplies ``raises``, ``approx``, ``fixture``, ``mark`` and the handful of
helpers the suite touches, and records ``parametrize`` cases so the runner can
expand them.

It is deliberately thin. Anything it cannot emulate it declines loudly rather
than quietly passing, so a test that depends on real pytest behaviour shows up
as an error instead of a false pass.
"""

from __future__ import annotations

import math
import sys
import types


class Approx:
    def __init__(self, expected, rel=None, abs=None):  # noqa: A002
        self.expected = expected
        self.rel = rel
        self.abs = abs

    def _close(self, a, b) -> bool:
        # Matches pytest.approx defaults: relative 1e-6, absolute 1e-12. The
        # absolute floor is what makes a comparison against exactly 0 work.
        rel = 1e-6 if self.rel is None else self.rel
        tol = 1e-12 if self.abs is None else self.abs
        return math.isclose(a, b, rel_tol=rel, abs_tol=tol)

    def __eq__(self, other):
        if isinstance(self.expected, (list, tuple)):
            if len(other) != len(self.expected):
                return False
            return all(self._close(a, b) for a, b in zip(other, self.expected))
        if isinstance(self.expected, dict):
            if set(other) != set(self.expected):
                return False
            return all(self._close(other[k], self.expected[k]) for k in self.expected)
        return self._close(other, self.expected)

    def __repr__(self):
        return f"approx({self.expected!r})"


def approx(expected, rel=None, abs=None):  # noqa: A002
    return Approx(expected, rel=rel, abs=abs)


class _Raises:
    def __init__(self, expected, match=None):
        self.expected = expected
        self.match = match
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise AssertionError(f"DID NOT RAISE {self.expected}")
        if not issubclass(exc_type, self.expected):
            return False
        if self.match is not None:
            import re
            if not re.search(self.match, str(exc)):
                raise AssertionError(
                    f"pattern {self.match!r} not found in {str(exc)!r}"
                )
        self.value = exc
        return True


def raises(expected, match=None):
    return _Raises(expected, match=match)


class Skipped(Exception):
    """Raised by skip(); the runner reports these separately from failures."""


def skip(reason: str = "", allow_module_level: bool = False):
    raise Skipped(reason)


def fail(reason: str = ""):
    raise AssertionError(reason)


def xfail(reason: str = ""):
    raise Skipped(f"xfail: {reason}")


def importorskip(name: str, minversion=None, reason=None):
    import importlib
    try:
        return importlib.import_module(name)
    except ImportError:
        raise Skipped(reason or f"could not import {name!r}") from None


def warns(expected, match=None):
    return _Raises(expected, match=match)


def fixture(*args, **kwargs):
    """Return the function unchanged; the runner skips tests that need one."""
    if len(args) == 1 and callable(args[0]) and not kwargs:
        args[0].__is_fixture__ = True
        return args[0]

    def deco(fn):
        fn.__is_fixture__ = True
        return fn
    return deco


def param(*values, **kwargs):
    return values[0] if len(values) == 1 else values


class _Mark:
    """``mark.anything`` is a no-op decorator; parametrize is recorded."""

    def parametrize(self, argnames, argvalues, **kwargs):
        def deco(fn):
            cases = getattr(fn, "__parametrize__", [])
            cases.append((argnames, list(argvalues)))
            fn.__parametrize__ = cases
            return fn
        return deco

    def __getattr__(self, name):
        def deco(*args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return lambda fn: fn
        return deco


mark = _Mark()


def install() -> None:
    """Register this module as ``pytest`` if the real one is absent."""
    try:
        import pytest  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    mod = types.ModuleType("pytest")
    for name in ("approx", "raises", "skip", "fail", "xfail", "fixture",
                 "param", "mark", "Skipped", "importorskip", "warns"):
        setattr(mod, name, globals()[name])
    mod.__version__ = "0.0-shim"
    sys.modules["pytest"] = mod
