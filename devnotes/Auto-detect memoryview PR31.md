# Auto-detect memoryview vs object return types (modelx-cython)

## Context

`CombinedCellsInfo.get_rettype_expr` ([builder.py:91-106](modelx_cython/builder.py:91)) emits a const memoryview return type (`const double[:]`) for every cells whose sampled return is a real-valued numpy array (gate at builder.py:95: `is_real_value and is_array_returned`). But formula bodies are emitted verbatim, so consumers that do more than element access break the generated Cython — e.g. CashValue_ME's `pv_net_cf` adds six memoryview-typed calls (examples/CashValue_ME_nomx/_mx_classes.py:1554-1559, its `_cy` output doesn't compile), and BasicTerm_S does `sum(list(...) * self.disc_factors()[:self.proj_len()])`. The current workaround is a hand-written spec override `{"cells": {"disc_factors": {"return_type": "object"}}}` (modelx_cython/tests/samples/basicterm_s/spec.py).

This change automates that decision with a **static libcst usage-classifier pass**: memoryview is emitted only when every model-internal use of the cells' return value is provably element access; otherwise fall back to `object`. Conservative = correct (worst case loses an optimization, never a broken build). No call-site rewriting, no runtime proxies (the sys.setprofile tracer is observe-only and cannot substitute values).

User-confirmed policy decisions:
- Cells with **no model-internal call sites** (only consumed by external scripts): **keep memoryview** (status quo; single-constant flip point).
- Add spec value **`"return_type": "memoryview"`** as an escape hatch forcing memoryview past the classifier.

## Design overview

1. **Phase split in cli.py**: parse + build ALL ModuleInfos first, then run the cross-module usage pass and annotate verdicts, then transform + write. (Today [cli.py:106-126](modelx_cython/cli.py:106) does everything per-module in one loop. `ModuleTransformer.transformed` and `PXDGenerator.code` are lazy, and nothing touches `get_rettype_expr` during ModuleInfo construction — verified — so the split is behavior-neutral.)
2. **New module `modelx_cython/usage.py`**: a dict-driven classifier core (unit-testable without builder objects) + thin production wrappers reading ModuleInfos.
3. **One gate change in builder.py**: all 6 transformer emission sites + pxd funnel through `get_rettype_expr`, so a single condition change is consistent everywhere (pxd cache attr, cdef `_f_`, cpdef public, annotations). `is_arrayable` requires `not is_array_returned` — no interaction.

## Changes by file

### 1. `modelx_cython/usage.py` (new)

Core (imports libcst, `parser.ParentScopeAddin`, consts only — no builder import in the core):

- `@dataclass UsageVerdict`: `fqname, ndim, only_element_access=True, has_internal_uses=False, unsafe_sites: List[str]` (module:line + reason, for INFO logging).
- `CellsResolver(cells_by_class, refs_by_class, spaces_by_class)` — plain-dict maps: class fqname → {public cells name → cells fqname}; class fqname → {ref/param name → mx_class fqname or ""}; class fqname → {child space attr → child class fqname}. `resolve_chain(cls, ["bar", "Qux", "qux"])` walks arbitrary-length attribute chains (real case: `self.bar.Qux.qux()` in the ref_space sample): child-space attr → new class; space-typed ref (non-empty `mx_class`) → new class; known non-space ref (`self.np`, tables, scalar params — empty `mx_class`) → `NOT_CELLS` (ignore); unknown base → `UNKNOWN` (poison by name). Final attr in `cells_by_class` → `RESOLVED(fqname)`. Child-space class fqname derivation mirrors transformer.py:376-377: `parent_pkg + "._m_" + cls_name[len("_c_"):] + "._mx_classes._c_" + attr`.
- `UsageVisitor(cst.CSTVisitor, ParentScopeAddin)` — `METADATA_DEPENDENCIES = (ScopeProvider, ParentNodeProvider, PositionProvider)`. Tracks the enclosing top-level `_c_*` class (same GlobalScope check as transformer.leave_ClassDef); `visit_Call` extracts a `self.`-rooted attribute chain and hands it to the classifier; always returns True so nested calls in args are visited. Chain roots that are a plain other Name, a Call, or a Subscript → UNRESOLVED-with-name (poison). Bare-name calls (`sum`, `len`, `range`) are ignored — cells are always attribute-accessed in nomx code. Public-name matching means the caching wrapper's `self._f_x()` never self-poisons.
- `UsageClassifier(candidates: {fqname: ndim}, resolver, array_returning: Set[str])` — `classify_module(fqname, wrapper)` re-visits the parser's existing `ModuleVisitor.wrapper` (no third parse; metadata providers cache). Verdict mutation is monotone (True→False), so cross-module aggregation is order-independent.

**Parent-chain walk** (the heart; from the Call node upward via ParentNodeProvider):

- Parent is `Subscript` with current node as its `.value`: inspect `SubscriptElement`s.
  - Any `Index` whose value is a Tuple/List/Set/Dict/Comparison/BooleanOperation/Ellipsis/None literal → UNSAFE (fancy indexing / newaxis). Otherwise Index values are **permissively** accepted (arbitrary int expressions like `[self.age(t), max(min(5, d), 0)]` must stay safe — BasicTerm_SC's `mort_table_array` works today and must not regress), except an index containing an embedded call resolving to an array-returning cells → UNSAFE.
  - All-Index, arity == remaining ndim → scalar element → **SAFE terminate**. (Parens are libcst node attributes, not parents, so `(1 + self.disc_rate_mth()[t])**(-t)` inside a genexp terminates correctly at the Subscript.)
  - Index count + slice count > remaining ndim → UNSAFE. Otherwise (partial index and/or any Slice) → still a memoryview: `remaining -= n_index`, move cursor to the Subscript, continue. (`self.mat()[0, :][j]` → safe; `self.arr()[:n] * list` → continues to BinaryOperation → unsafe — the disc_factors pattern.)
- Parent is `cst.Expr` (bare statement, value discarded) → SAFE.
- **Anything else → UNSAFE** (default-deny): Attribute (`.sum()`), BinaryOperation (incl. `@`), BooleanOperation, Comparison, UnaryOperation, `Arg` of any call (`len`, `max`, `np.*`), Return, Assign/AnnAssign/AugAssign/NamedExpr (aliasing — v1 conservative, ScopeProvider access-chasing is an explicit future refinement), collection Elements, comprehension elt/iterable, FormattedString, IfExp, Starred.

Production wrappers (only builder-coupled code): `collect_candidates(module_infos)` → `{fqname: ndim}` for cells with typeinfo, array-returned, real-valued, ndim ≥ 1, no spec `return_type`, not special; plus `array_returning` set (any dtype). `build_resolver(module_infos)` from `ClassInfo.cells` / `ClassInfo.refs` (`CombinedRefInfo.mx_class`; includes itemspace params via `_add_space_params`) / `ClassInfo.spaces`. `analyze_usage(units)` runs the classifier over every module. `apply_verdicts(module_infos, verdicts)` sets `cells.usage = verdict` and logs one INFO line per demotion (style of builder.py:292).

### 2. `modelx_cython/config.py`

Add `RET_MEMORYVIEW = "memoryview"` to `TransSpec` (next to `RET_T`, config.py:32). `typedefs.str_to_type` untouched — "memoryview" is intercepted in builder before that lookup.

### 3. `modelx_cython/builder.py`

- Module constant `MEMORYVIEW_FOR_EXTERNAL_ONLY_ARRAYS: bool = True` — the single flip point for the no-internal-uses policy (user chose status quo True).
- `CombinedCellsInfo`:
  - Class attr `usage: Optional[UsageVerdict] = None` (set per-instance in phase 2; `TYPE_CHECKING` import, no runtime cycle).
  - `__init__`: `self._force_memoryview = (self._spec_ret_t == TransSpec.RET_MEMORYVIEW)`; when set and typeinfo exists but the return is not a real-valued array → `ValueError("...'memoryview' requires a real-valued numpy array return...")`; when set but no typeinfo → warning, ignored (falls to object, consistent with builder.py:66).
  - `norm_type` (builder.py:57-64): spec branch becomes `if self._spec_ret_t and not self._force_memoryview:` so "memoryview" falls through to the sampled type.
  - New **plain `@property`** (NOT cached — `usage` is mutated between phases) `use_memoryview`: `_force_memoryview` → True; `usage is None` → True (legacy behavior for anything the analysis didn't cover); no internal uses → `MEMORYVIEW_FOR_EXTERNAL_ONLY_ARRAYS`; else `usage.only_element_access`.
  - `get_rettype_expr` (builder.py:95-102): inside the existing `is_real_value and is_array_returned` branch, wrap the memoryview emission in `if self.use_memoryview:` with `else: return "object"` (the literal string — `typ` there is the scalar element type).
  - Small `@property ret_ndim` returning `self._rt.ret_type.ndim` for `collect_candidates`.

### 4. `modelx_cython/cli.py`

Restructure [main_handler:106-126](modelx_cython/cli.py:106) into three sub-phases; everything else (backups, copytree, `_mx_sys.pxd` copy, `run_sample`, spec load, `create_setup`, compile branches) stays byte-identical:

- Phase 1: loop over `logger.modules`, compute all paths, read source, build `ModuleVisitor` + `ModuleInfo`, collect into a small `_TransUnit` dataclass (fqname, source, visitor, module_info, 4 paths).
- Phase 2: `verdicts = analyze_usage(...)`; `apply_verdicts(...)`.
- Phase 3: loop over units — `ModuleTransformer(u.source, u.module_info)`, `PXDGenerator`, write `.py`/`.pxd`/`__init__.pxd`, append to `modules` in the same order as today (identical setup.py output). Constructing the transformer in phase 3 structurally guarantees no type expression renders before verdicts are set.

### 5. Tests

- **`modelx_cython/tests/test_usage.py` (new)** — plain pytest against the dict-driven core (style of test_tracer.py), ndim injected via the `candidates` mapping; parametrized use-forms: subscript-only safe; genexp scalar-termination safe (real `_f_disc_factors` shape); slice-then-operate unsafe; ndim=2 full index safe / partial-index-then-op unsafe / chained `[i][j]` safe / mixed `[0, :][j]` safe; whole-array arithmetic, len(), returned whole, `.mean()`, comprehension elt, fancy index (List/Tuple), aliased-to-local — all unsafe. Discrete tests: no-internal-uses verdict; cross-space resolution via refs' mx_class and via 3-level `self.bar.Qux.arr()` chain; `self.np.array(...)` ignored (known non-space ref, no poisoning); unresolvable base poisons matching cells names only; array-valued-index unsafe; cross-module aggregation order-independence; non-`_c_` classes ignored; double-visit of one MetadataWrapper (libcst canary — libcst is unpinned).
- **New sample `modelx_cython/tests/samples/array_usage`** (model `ArrayUsage`, layout modeled on various_types; no lifelib needed, real Cython build): space `Data` with cells isolating each verdict — `arr_elem` (int-subscript consumer → stays `const double[:]`), `arr2d` (`[i, 1]` → stays `const double[:, :]`), `arr_whole` (`(arr_whole() * 2.0).sum()` → object), `arr_sliced` (`sum(list(...) * arr_sliced()[:3])` → object), `arr_cross_elem`/`arr_cross_whole` (consumed from space `Consumer` via ref → memoryview/object), `arr_external` (no internal uses → policy: memoryview). New `test_array_usage` in test_samples.py (pattern of `test_varying_integral_arg_types` incl. pxd-text assertions like test_samples.py:163) asserting the six `_v_*`/cpdef types, then `assert_cy.py` numeric equality. Spec variants run with `--translate-only` (text assertions only, no C build): `spec_force_mv.py` forces `arr_whole` back to memoryview ("memoryview" wins over classifier); `spec_bad_mv.py` puts "memoryview" on a scalar cells → nonzero returncode + error text.
- **basicterm_s end-to-end proof**: remove the `disc_factors` `return_type: "object"` override from modelx_cython/tests/samples/basicterm_s/spec.py (keep `cells_params`); add a guarded assertion `"cdef object _v_disc_factors\n" in pxd` in `test_mx2cy_with_lifelib`. The existing lifelib test then compiles and numerically validates the classifier on a real model. Land in the same PR as the classifier.

## Decisions and known limitations (documented, not blocking)

- Conservative by design: any `Arg` use is unsafe even though many np calls work on memoryviews via the buffer protocol — future whitelist is the natural relaxation point. Aliasing → unsafe in v1.
- Only modules in `logger.modules` (traced) are classified/transformed — same coverage as today's translation; untraced consumers were equally broken before, so no regression.
- Index int-ness remains unproven (pre-existing; verbatim transformer). ndim=0 arrays excluded from candidates (pre-existing `get_rettype_expr` oddity, out of scope).
- Expected outcomes: BasicTerm_S — `disc_rate_mth` stays memoryview, `disc_factors` → object (reproduces the manual override automatically); BasicTerm_SC — all `_c_Data` array cells stay memoryviews; CashValue_ME — `disc_factors` and the `pv_*` family → object (fixes its non-compiling output). With policy True there is **zero generated-code diff** for every existing test sample (various_types' array cells have no internal consumers → policy branch → unchanged).

## Implementation order

1. config.py: `RET_MEMORYVIEW`.
2. builder.py: `usage` attr, `_force_memoryview` + validation, `norm_type` tweak, `use_memoryview` property, gate change, `ret_ndim`, policy constant.
3. cli.py two-phase restructure with a temporary `analyze_usage` stub returning `{}` — proves the restructure is behavior-neutral against the existing suite.
4. usage.py core + `test_usage.py`.
5. usage.py production wrappers; swap the stub for the real pass.
6. array_usage sample + `test_array_usage`.
7. basicterm_s spec change + pxd assertion.

## Verification

Local PC (this machine): run with `C:/Users/fumito/anaconda3/envs/py313/python.exe -m pytest modelx_cython/tests -x -q` (needs modelx, lifelib, Cython, a C compiler; lifelib-dependent basicterm tests are the slowest — run `test_usage.py` and `test_array_usage` first during development). Confirm: full suite green; `test_array_usage` pxd assertions; basicterm_s passes **without** its disc_factors override; log output shows one INFO line per demoted cells.