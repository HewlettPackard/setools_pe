"""
Shared fixtures and module-loading helpers for the setools_pe test suite.

The three scripts under test (se_log_analyser, se_check_type, se_policy_merger)
have no .py extension, so they are imported via importlib machinery.  This
conftest centralises that boilerplate so every test file can simply import the
modules it needs.
"""

import importlib
import importlib.machinery
import importlib.util
import os
import sys

# ── Base directory of the project (parent of tests/) ────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


def _load_script_module(name: str):
    """Import a script without .py extension as a proper Python module."""
    script_path = os.path.join(_PROJECT_ROOT, name)
    loader = importlib.machinery.SourceFileLoader(name, script_path)
    spec = importlib.util.spec_from_loader(name, loader, origin=script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── Pre-loaded modules (imported once, reused across test files) ────────────
se_log_analyser = _load_script_module("se_log_analyser")
se_check_type = _load_script_module("se_check_type")
se_policy_merger = _load_script_module("se_policy_merger")
