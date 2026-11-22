from libc.stdint cimport uint64_t
from libc.stddef cimport size_t

ctypedef uint64_t word
cdef struct Entry:
    word a
    word b
    word c
    word value
    bint ready

cdef struct PrimaryEntry:
    word a
    word b
    word c
    word value
    unsigned char state

cdef union Bits:
    double real
    word integer


# Inline in the pxd so compiled model modules inline the key and value conversions (D74).
cdef inline word int_word(long long value) noexcept:
    return <word>value


cdef inline word value_word(double value) noexcept:
    cdef Bits bits
    bits.real = value
    return bits.integer


cdef inline word double_word(double value) noexcept:
    return value_word(value if value != 0.0 else 0.0)


cdef inline double word_double(word value) noexcept:
    cdef Bits bits
    bits.integer = value
    return bits.real


cdef inline long long word_int(word value) noexcept:
    return <long long>value

cdef class NumericCache:
    cdef Entry* entries
    cdef size_t capacity, used
    cdef PrimaryEntry* primary
    cdef size_t primary_capacity, primary_used
    cdef bint get(self, word a, word b, word c, bint reflexive, word* value) except? -1
    cdef void put(self, word a, word b, word c, bint reflexive, word value) except *
    cdef bint reserve(self, word a, word b, word c, bint reflexive, Py_ssize_t* slot,
                      word* value) noexcept
    cdef void publish(self, word a, word b, word c, Py_ssize_t slot, word value) except *
    cpdef reset(self)

