"""Test suites build with strict lowering: an internal error of a table-read recogniser fails the build (T-5c)."""
import os

os.environ.setdefault("MX_CY_STRICT_LOWERING", "1")
