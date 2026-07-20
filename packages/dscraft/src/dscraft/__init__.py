"""DSCraft: a unified, MIT-licensed, local-first ML tooling platform.

This is the top-level ``dscraft`` package. Its subpackages (``dscraft.core``,
and eventually ``dscraft.automl``, ``dscraft.clean``, ``dscraft.forecast``,
``dscraft.graph``, ``dscraft.vision``, ``dscraft.tune``, ``dscraft.security``,
``dscraft.agent``) are the real implementations, not re-export shims --
``dscraft`` is one real, installable package rather than a pile of
independently-versioned packages.

``dscraft.core`` is the only subpackage with a hard, always-installed
dependency (``opentelemetry-api``); every other subpackage is expected to be
optional-extras-gated once it lands, so this top-level module deliberately
does not eagerly import any subpackage here.
"""

from __future__ import annotations

from importlib.metadata import version

__version__ = version("dscraft")

__all__ = ["__version__"]
