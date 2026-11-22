"""D116: one construction and one release of a top-level item, shared by collect and the direct route.

Collect's reused root and a directly created item are the same thing: ``C(base)`` with the base's
References and itself as its only root. Collect builds one per call and resets it per policy; the
direct route builds one per key. ``__delitem__`` releases the item it drops through one shared function. Releasing breaks the item's references to itself (``_space``,
``_mx_roots``) and its Cell objects, so it is freed by its reference count.
"""
import libcst as cst

from .consts import CY_MOD


def new_root(cls_name):
    return f'_mx_new_root_{cls_name}'


def release_root(cls_name):
    return f'_mx_release_root_{cls_name}'


def release_item(cls_name):
    return f'_mx_release_item_{cls_name}'


def functions(cls_name, contexts=()):
    """Module-level C functions for one compiled ItemSpace class.

    ``contexts`` are the class's per-instance lookup contexts (``native_contexts``). Every root takes
    them over privately (``_mx_enable_private_lookup``: warm lookups without the critical section, the
    round memo): items of an exported ItemSpace class are partitioned by the caller, the one documented
    thread rule for items that ``numeric_cache.exclusively_owned`` already relies on (audit-routes P6,
    D126). A context whose provider has no such method is left shared."""
    enable = "".join(
        f"    _mx_enable = getattr(root.{name}, '_mx_enable_private_lookup', None)\n"
        f"    if _mx_enable is not None:\n"
        f"        _mx_enable()\n" for name in sorted(contexts))
    return cst.parse_module(
        f"@{CY_MOD}.cfunc\n"
        f"def {new_root(cls_name)}(base: {cls_name}) -> {cls_name}:\n"
        f"    root: {cls_name} = {cls_name}(base)\n"
        f"    root._mx_copy_refs(base, base)\n"
        f"    root._mx_roots.append(root)\n"
        f"{enable}"
        f"    return root\n"
        f"\n\n"
        f"@{CY_MOD}.cfunc\n"
        f"def {release_root(cls_name)}(root: {cls_name}) -> None:\n"
        f"    root._mx_cells.clear()\n"
        f"    root._mx_roots.clear()\n"
        f"    root._space = None\n"
        f"\n\n"
        f"@{CY_MOD}.cfunc\n"
        f"def {release_item(cls_name)}(removed: object) -> None:\n"
        f"    # An item dropped from its ItemSpace (``del``, whole-model clear): released as above when it\n"
        f"    # is a compiled item without child Spaces or items of its own; anything else is left as is.\n"
        f"    item: {cls_name}\n"
        f"    if type(removed) is {cls_name}:\n"
        f"        item = {CY_MOD}.cast({cls_name}, removed)\n"
        f"        if not item._mx_spaces and not item._mx_itemspaces:\n"
        f"            {release_root(cls_name)}(item)\n").body
