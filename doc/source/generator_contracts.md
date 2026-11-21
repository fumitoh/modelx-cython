# Generator and reduction contracts

The regional integration experiment retains the normal modelx-cython Space/Cell
structure and exporter `b8b7472`. Qualification belongs to the exact source pins
and evidence receipts; it does not imply core promotion.

Declared helper capabilities use the existing CST and type facts.
`NativeReductions` is active for its proved numeric domains; general generators
remain ordinary object-capable Cython. There is no second syntax frontend,
formula interpreter or dataframe interpreter. Unsupported domains and replaced
helper callables retain their original generator and public Cell calls.

## Expression vocabulary

Use one namespace: `from modelx_helpers import expressions as e`. The facade re-exports
the exact registered function objects and adds no wrappers or second registry.
Input preparation belongs to `modelx_helpers.pandas_inputs`; pandas output construction
is an explicit boundary. The detailed live/native provider contract is in
modelx's `doc/source/helpers/EXPRESSION_STREAMS.md` and `NATIVE_TABLES.md`.

| Expression | Meaning |
|---|---|
| `e.periods(n)` or `e.periods(start, stop, step)` | Reusable range domain, with ordinary integer/index conversion and zero-step behavior. |
| `e.over(source, function=omitted, where=None, *, unpack=None)` | One-shot lazy map/filter producer. Acquire the source iterator eagerly; unpack each candidate, test its predicate, then evaluate its payload. |
| `e.flatten(sources)` | Lazy ordered concatenation; acquire inner iterators only as reached. |
| `e.zip_values((first, second), *, strict=False)` | Dedicated resumable zipped producer over an authored exact tuple of sources; ordinary zip acquisition, advancement, strictness, retries and retained-row lifetimes. |
| `e.sum_float(source, function=omitted, where=None) -> float` | Float-normalize each passing contribution lazily, then use float-start summation; empty input is float `0.0`. |
| `e.sum_values(source)` | Ordinary sum, including integer zero, arbitrary integers and mixed/custom arithmetic. |

```python
e.sum_float(e.periods(proj_len()),
            lambda t: net_cf(t) * discount(t))

all(e.over(e.periods(proj_len()),
           lambda t: abs(residual(t)) <= tolerance))

sum(e.over(e.zip_values((loss_days, probabilities)),
           lambda days, probability: days * probability, unpack=2))

next(e.over(e.zip_values((ages[:-1], ages[1:])),
            lambda lower, upper: (lower, upper),
            where=lambda lower, upper: lower <= age <= upper, unpack=2))

sum(e.flatten(e.over(e.periods(1, 6), lambda grade:
    e.over(e.periods(1, waiting_months() + 1),
           lambda month: pols_waiting(t, grade, month)))))
```

Typed shared lookups are `lookup_float`, `lookup_int`, `lookup_bool`,
`lookup_str` and `lookup_object`. `keys`, `row`, `column`, `group_sum` and
`to_pandas` compose with the same namespace; their preparation, scalar, missing,
alignment and public-output contracts belong to the pandas provider.

`sum_float` means precisely `sum((float(value) for value in values), 0.0)`
under the canonical builtin bindings. Optional mapping/filtering precedes that
normalization. It differs from ordinary `sum(values)` and from the older
`sum_f64(values) == sum(values, 0.0)` for integer prefixes, NumPy/custom numbers,
Decimal and empty-result types. These names must not be silently aliased.
A float-looking sample does not justify changing a model's ordinary sum.
Regional source adjustments retain the standard consumers.

The registered `ordinary_sum` capability permits a narrower physical path only
at an already-double Cell return boundary when every yielded contribution is
proved double. It reuses the existing version-qualified compensated `F64Sum`
state. Empty integer zero crosses the same existing double return conversion.
Intermediate assignments, explicit object-return Cells and unproved integer,
mixed or custom yields keep ordinary summation. No formula-length threshold or
product setting selects this path.

Numeric reductions admit the established range/singleton domains and immutable
literal tuples. Newly admitted tuple labels remain Python objects and iterate
in order; they are not encoded into a new category schema or unrolled by size.
The canonical callable guard, lazy generator fallback, public cached Cell
dispatch, subclass overrides and generator exception chaining remain intact.

## Identity traversal and raw result declarations

The ABI 2 stream provider also supports `e.over(source)` and where-only
filtering. It owns a fresh identity generator, even if the source is suspended;
send values are ignored and throw/close do not forward upstream. Unpacking
requires an explicit mapper. Explicit `function=None` remains a non-callable
mapper and fails lazily. Native packages require a fresh rebuild after this
optional-parameter C signature change.

```python
sum(e.over(net_cf(t) * discount(t) for t in e.periods(proj_len())))
```

Cython compiles the original generator body; no mapper callback or body
recognizer is introduced. `periods` keeps Python range progression for signed
limits, nonunit steps and wide integers. This syntax has provider/protocol and
normal CLI evidence; its all-model source materialization remains a separate,
unqualified candidate at the handover checkpoint.

For a Cell that must retain a raw NumPy scalar/object identity, the existing
public translation specification supports `return_type: "object"`. Declare
identity-preserving forwarding Cells as object too. The normal CLI regression
checks raw scalar identity, warnings, cache hits and pickle using direct function
References. Concrete `-> object` annotations are also recognized. String
annotations are not evaluated, object declarations do not propagate through
forwarding Cells, and an explicit translation specification takes precedence.
Whole-model pickle with module References remains a separate baseline limitation.

## Evaluation and protocol

Consumer callables and arguments retain authored left-to-right evaluation.
Outer iterator acquisition is eager; payloads, filters and dependent inner
bounds stay lazy. An empty outer producer never evaluates an inner bound.
`unpack=N` is explicit because a tuple can itself be one intended argument;
flat arities two, three and four use bounded unpacking and the original binding
order before predicates or payloads.

`over` and `flatten` retain standard send/throw/close behavior. Cython owns their
generator execution and exception normalization. A thin protocol preflight
corrects observed Cython 3.2.9 rejected-send/invalid-throw state differences;
it does not construct a second exception or formula interpreter. Source
exhaustion yields plain `StopIteration`; a body/predicate/unpack/inner-acquisition
`StopIteration` becomes `RuntimeError('generator raised StopIteration')` with
the original cause/context. Other exceptions preserve identity. A retained
producer stays resumable after a consumer-side conversion failure.

The zipped producer is an iterator rather than a generator frame. Source
exceptions, transient exhaustion and strict mismatches do not automatically
close it. Strict truth conversion precedes ordered iterator acquisition.
Aliased iterators share advancement; two uses of a sequence get independent
cursors. Cached result-row ownership matches builtin zip. Native free-threaded
3.13 builds are rejected because its supported API cannot establish the required
unique tuple ownership; 3.14 Windows qualification is scoped to exact-pin receipts.

Standard `all`, `any`, `min`, `max`, `next`, `sum`, `sorted`, `tuple`, `list`,
`set` and `dict` keep their own short circuit, empty, identity, first-tie, numeric
and materialization contracts. Eager comprehensions remain eager. The generic
argument-preservation pass retains selected scalar classes for object-result
extrema and Python-number semantics for unproved powers. It does not promise
the original order of exceptions. Remaining unproved powers include integer
zero powers and negative exponents;
the qualified local proved-real power mechanism remains unchanged.

Exact CPython generator type/frame/code introspection is outside the inventoried
pricing contracts. Arbitrary internal helper-global changes require rebuilding;
ordinary current helper Reference/module-member replacements still use the
registered callable identity check and authored positional/keyword call shape.
A surrounding traced Cell retains its independently documented result/cache ABI.

## Compiler and provider mechanism

`helpers.CAPABILITIES` is the sole registry. Metadata declares provider operation,
physical result, supported iteration span operations and keyword layout.
`helper_members` resolves direct and module References without invoking module
descriptors. The binder reads declared signatures, checks lexical receiver
identity and emits direct Cython calls. It neither inspects callback bodies nor
infers a numeric domain from formula text. Ordinary Cython compiles callbacks.

The same binder runs in ordinary C-function formulas and retained compiled-def
paths for captured defaults, yield and uncached closures. Keeping a method as
a compiled def preserves Cython's closure/default handling without requiring
its registered helpers to execute Python implementations. Public signatures,
qualified captured-default guards, enums and cache architecture are preserved.

Canonical range sources within signed 64-bit bounds use overflow-safe C
progression state, including descending/nonunit ranges and the minimum signed
step. Exact tuple/list sources use native index state. Wider numeric domains
and arbitrary iterators retain semantic compatibility; that compatibility is
not automatically complete native coverage. `sum_float` uses the shared
compensated double summation state and an object dispatcher for replacements.
No PGO, forced-inlining flags or product/Cell-specific compiler branches apply.

## Source and qualification evidence

The offline, guarded CST authoring adjustment converts 520 regional generator
expressions (521 clauses) to `over`/`flatten`, including thirteen zipped sources.
It retains before/after source hashes, ordinary consumers, eager comprehensions,
original powers and all model inputs. Unsupported escaping closures, assignment
expressions, nested generator objects, asynchronous clauses and unsafe target
bindings are declined. This source editing utility is not a compiler frontend.

Native completion requires each original creation/consumption/indirect-helper
operation to link to its source adjustment, semantic tests, loaded extension,
generated C body and concrete/transitive producer admission. Source profiling,
full original financial comparisons and additional default/diagnostic probes
are separate evidence. Successful compilation, a C file, matching financial
values or a compatibility branch alone cannot close a native row. Remaining
pandas producers and unobserved/dynamic targets stay explicit until resolved.
Matched same-VM performance follows correctness and process quiescence. Windows
acceptance and user-approved core promotion remain separate gates.
