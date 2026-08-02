# Module reference

This section documents the modules of the `modelx_cython` package.

```{note}
modelx-cython is primarily used through the {doc}`mx2cy <../cli>` command
line tool.  The Python API documented here is internal and unstable; it is
provided for contributors and advanced users.
```

The modules are listed in pipeline order — the order in which they act
when `mx2cy` translates a model (see {doc}`../architecture`):

```{toctree}
:maxdepth: 1

modelx_cython
cli
tracer
monkeytype_tracing
parser
builder
usage
transformer
config
typedefs
consts
```
