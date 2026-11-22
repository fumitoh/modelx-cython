# Native providers for Cell results

A Python class can declare a native representation for a Cell that returns its
instances. The Python class remains the semantic owner and works without Cython
or a compiled provider. Declare the protocol version, provider resource name,
provider ABI, and semantic type identity on the exact owner class.

```python
class Curve:
    __modelx_cython_type__ = (1, 'example.rates', 'rates-v1', 'example.Curve')
```

Put `rates.provider.json` beside `rates.pyx` and `rates.pxd` in the `example`
package. The manifest connects the owner to the provider's Cython types and
symbols. Provider symbol names are local to the package.

```json
{
  "protocol": 1,
  "provider": "example.rates",
  "abi": "rates-v1",
  "types": {
    "example.Curve": {
      "view_type": "CurveView",
      "bind": "bind_curve",
      "members": {
        "rate_at": {"signature": "int64_label_to_float64", "native": "read_rate"}
      }
    }
  }
}
```

The `.pxd` declares `bind_curve(object) -> CurveView` and
`read_rate(CurveView, long long) -> double` as Cython functions. The compiled
provider publishes `__modelx_cython_provider__ = (1, 'example.rates', 'rates-v1')`.
At model import, the generated package checks that declaration and a fingerprint
of the provider manifest, implementation and `.pxd` before loading native Cells.

The binder retains the owner for the lifetime of the view. For a local view
received through a forwarded Cell, it returns the same view. For a foreign
wrapper, it can obtain the Python owner through `__modelx_cython_owner__()`,
validate the exact owner type, and create a view local to the current model.
Providers should expose the owner through the same method for callers who need
the Python representation. The Python-facing view must also support the owner
methods a formula may call through its ordinary object path. The binder raises
a clear error when a typed Cell returns a value outside the declared domain.

The compiler uses a consistent exact owner type observed in the sample to bind
the result. An explicit `object` return annotation or `return_type` setting
keeps the Cell on the ordinary result path. A missing optional manifest also
keeps the ordinary path. A view can declare `"admission": "declared_or_forwarded"`
to require an explicit owner return annotation or a resolved forwarding Cell
with the same provider, ABI and semantic type. The default is `"observed"`.
Set `"bind_cell": true` when the binder takes the Cell's qualified name as a
second argument for diagnostics.

Providers that use the Cython view methods directly can omit `members`. Each
additional capability is an opt-in to its own lowering.

The `int64_label_to_float64` member signature
lowers a direct member call on a resolvable typed Cell result when the label is
a signed 64-bit integer literal. Other arguments use the Python-facing method.
A provider can also declare `__getitem__` with the
`string_label_to_object` signature for a string-literal item read from a
Cell result in an assignment. This operation takes the owner and the string
and returns a Python object. An owner that uses this operation may omit the
native view and binder.

## Bound numeric methods

A typed view can declare `numeric_table` to bind numeric reads on the
private receiver used by native `collect`. The `methods` keys are the
Python-facing method names. Each entry names its own native symbols.

```json
{
  "owner_field": "owner",
  "ready": "prices_ready",
  "methods": {
    "price": {
      "signature": "lookup_f64",
      "read": "bound_price",
      "prepare": "prime_price"
    }
  }
}
```

The view exposes `owner_field` as a Cython object field holding the Python
owner. The provider declares a default-constructible `LookupContext`
extension type with a Cython `bint _private_owner` field. The field starts
false. The optional `_mx_enable_private_lookup()` hook enables private
binding when `collect` owns the receiver. The ordinary view method remains
the fallback for lookup outside this context.

The `ready` function takes `(LookupContext, const void* owner,
const void* column) -> bint noexcept`. It checks owner identity and either
column identity for a named column or a null pointer for a numeric column.
The context retains the owner and any column or prepared state used by the
bound read. The `read` function takes `(LookupContext, long long key,
long long column, bint named, bint has_default, double default) -> double`.
The `prepare` function takes `(LookupContext, View, long long key,
object column, object default=*) -> double`. It prepares state and performs
the initial lookup. For a `previous_f64` method, omit the default parameter
from `prepare` and declare `read_integer(View, long long, long long)` and
`read_object(View, long long, object)` symbols, both returning `double`.
These implement the ordinary previous-key lookup for the two column types.

Function helpers can opt into a namespace-aware ready check by adding its
symbol as the fourth entry in `__modelx_cython_bound_numeric__`. That check
takes the context, owner and column identity pointers, and the current helper
callable. It must validate the callable before a bound read can reuse a value.

Lookup methods may declare `read_text` for a string key and a literal column.
It takes the context, typed view, string key, and string column and returns
a `double`. A table may also declare `bounds`. Its `first` and `last` name
registered Python bound helpers, and `index` and `index_max` name the index
attribute and its upper-bound method. The `bounded` and `clamped` symbols
implement the corresponding proven read shapes with a `double*` output.
They return a `bint` indicating whether the native operation supplied a value.
When the native operation declines, the generated formula retains its original
lookup and bounds operations.

## Numeric Record fields

A Record can declare `numeric_record` with the `lookup`, `read_field`,
`read_cast`, `field_ready`, `cast_ready`, and `bound_read` symbols. The
compiler uses `read_cast` for `float(record["column"])` when the Record also
declares the string-label `__getitem__` member. It uses `read_field` for
a two-argument call to a Python helper registered with this provider and
the `lookup` operation. Unregistered calls keep their existing dispatch.

Both read functions take the current Python callable, `LookupContext`,
the Record owner, and the string key. `read_field` also takes an optional
helper namespace. They return a `double` and preserve the current callable's
behavior. The ready functions take the context, owner and key identity
pointers, and the current namespace or conversion callable. They return
`bint noexcept`. The context must retain all owners and values needed for
`bound_read(LookupContext) -> double` until the binding is replaced.

## Grouped numeric reads

A Python owner can declare `grouped_table` without defining a native view.
Its `slice_method`, `index_field`, and `lookup_method` entries identify the
ordinary grouped read. The compiler recognizes membership checks for both
the group and the selected table key before a capped double result.
An observed provider can participate through a direct Reference or a Cell
result in a locked Space. The original methods remain the fallback.

The declaration supplies a `probe(LookupContext, object owner, long long
group, str column, Py_ssize_t* slot) -> object` symbol. It returns an owned
prepared view and writes a nonnegative group slot, `-1` for an absent group,
or `-2` to request the ordinary path. The `read(LookupContext, object
prepared, Py_ssize_t slot, long long key, double* value) -> int` symbol
returns `0` for an absent key, `1` after writing the value, or `2` for the
ordinary path. Exceptions propagate. The probe checks the exact owner
type and the view keeps the prepared state alive across key evaluation.

Providers can also declare both `bound_probe` and `bound_read` for a stable,
collect-owned context. The probe takes `(LookupContext, long long group,
Py_ssize_t* slot) -> bint` and returns true only if the binding is valid.
The read takes `(LookupContext, Py_ssize_t slot, long long key,
double* value) -> int` and follows the same status convention. The context
may reuse this state only under the private ownership contract.

## Reusing compiled provider binaries

Set `MODELX_CYTHON_PROVIDER_CACHE` to a build-cache directory to reuse a native
provider when another build has the same extension identity, source and
declaration dependencies, Python ABI, Cython version, compiler and flags. The
build copies the provider binary into the generated model package. The model
loads that local binary when the build cache is unavailable. Distinct qualified
extension names have distinct cache entries.
