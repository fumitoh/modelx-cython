"""diag-run8: read the fast path of a generated C function (from its definition to a marker of its
fallback, taken from the source-line comments Cython writes) and list its reference-count operations."""
import re

REFCOUNT = re.compile(r"\b(__Pyx_INCREF|__Pyx_DECREF|__Pyx_XDECREF|__Pyx_XINCREF|Py_INCREF|Py_DECREF|"
                      r"Py_XDECREF|Py_XINCREF|__Pyx_GOTREF|__Pyx_GIVEREF|__Pyx_NewRef)\b")


def fast_path(c_text, function, fallback_marker=None):
    """Lines of the definition of the C function whose name ends with ``function`` up to the first
    source comment containing ``fallback_marker`` (the whole body when None)."""
    lines = c_text.splitlines()
    start = next(i for i, line in enumerate(lines)
                 if re.match(rf"^static .*\b\w*{re.escape(function)}\(", line) and line.rstrip().endswith("{"))
    body = []
    for line in lines[start:]:
        if fallback_marker is not None and fallback_marker in line:
            break
        body.append(line)
        if line.startswith("}"):
            break
    return body


def refcount_operations(c_text, function, fallback_marker=None):
    return [line.strip() for line in fast_path(c_text, function, fallback_marker) if REFCOUNT.search(line)]
