from libc.stddef cimport size_t
from libc.stdint cimport uint64_t

ctypedef uint64_t word

cdef union WideKeyBits:
    double real
    word integer

cdef inline word key_word(double value) noexcept:
    cdef WideKeyBits bits
    bits.real = value if value != 0.0 else 0.0
    return bits.integer

cdef inline word mix(word value) noexcept:
    value ^= value >> 30
    value *= <word>0xbf58476d1ce4e5b9
    value ^= value >> 27
    value *= <word>0x94d049bb133111eb
    return value ^ (value >> 31)

cdef inline word hash_step(word state, word key, size_t offset) noexcept:
    return state ^ mix(key + (<word>offset + 1) * <word>0x9e3779b97f4a7c15)

cdef class WideNumericCache:
    cdef word* entries
    cdef size_t width, stride, capacity, used
    cdef bint private_owner
    cdef bint get(self, const word* key, bint reflexive, word hash_value, word* value) except? -1
    cdef void put(self, const word* key, bint reflexive, word hash_value, word value) except *
    cdef bint reserve(self, const word* key, bint reflexive, word hash_value,
                      Py_ssize_t* slot, word* value) noexcept
    cdef void publish(self, const word* key, word hash_value, Py_ssize_t slot, word value) except *
    cpdef reset(self)
