from __future__ import annotations

import importlib


def test_frontend_module_imports():
    mod = importlib.import_module("frontend.app")
    assert isinstance(getattr(mod, "BACKEND_URL", ""), str)



