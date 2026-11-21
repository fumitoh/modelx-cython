"""One object-preserving replacement-call ABI for registered providers."""

cdef inline object forward_call(object function, tuple keywords, tuple names,
                                 tuple values, object omitted_argument):
    cdef list positional = []
    cdef dict named = {}
    for name in keywords:
        named[name] = values[names.index(name)]
    for name, value in zip(names, values):
        if value is not omitted_argument and name not in named:
            positional.append(value)
    return function(*positional, **named)
