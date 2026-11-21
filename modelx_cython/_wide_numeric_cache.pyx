# cython: freethreading_compatible=True
"""Exact native keys for numeric Cells with wider fixed signatures."""
from libc.stdlib cimport calloc, free
from libc.string cimport memcmp, memcpy, memset
cimport cython


cdef inline word key_hash(const word* key, size_t width) noexcept:
    cdef word hash_value = 0
    cdef size_t i
    for i in range(width):
        hash_value = hash_step(hash_value, key[i], i)
    return hash_value


cdef inline word* position(word* entries, size_t capacity, size_t width, size_t stride,
                           const word* key, word hash_value) noexcept:
    cdef size_t index = hash_value & (capacity - 1)
    cdef word* row = entries + index * stride
    while row[width + 2] and (row[width + 1] != hash_value or memcmp(row, key, width * sizeof(word))):
        index = (index + 1) & (capacity - 1)
        row = entries + index * stride
    return row


cdef inline void grow(WideNumericCache cache) except *:
    cdef size_t i, capacity = cache.capacity * 2
    cdef size_t row_bytes = cache.stride * sizeof(word)
    cdef word *grown
    cdef word *row
    if cache.capacity > ((<size_t>-1) // row_bytes) // 2:
        raise MemoryError("numeric cache exceeds addressable storage")
    grown = <word*>calloc(capacity, row_bytes)
    if grown == NULL:
        raise MemoryError()
    for i in range(cache.capacity):
        row = cache.entries + i * cache.stride
        if row[cache.width + 2]:
            memcpy(position(grown, capacity, cache.width, cache.stride,
                            row, row[cache.width + 1]), row, row_bytes)
    free(cache.entries)
    cache.entries = grown
    cache.capacity = capacity


cdef inline bint read(WideNumericCache cache, const word* key, word hash_value,
                      word* value) noexcept:
    cdef word* row = position(cache.entries, cache.capacity, cache.width, cache.stride, key, hash_value)
    if not row[cache.width + 2]:
        return False
    value[0] = row[cache.width]
    return True


cdef inline void write(WideNumericCache cache, const word* key, word hash_value,
                       word value) except *:
    cdef word* row
    if cache.used * 2 >= cache.capacity:
        grow(cache)
    row = position(cache.entries, cache.capacity, cache.width, cache.stride, key, hash_value)
    if not row[cache.width + 2]:
        cache.used += 1
        memcpy(row, key, cache.width * sizeof(word))
        row[cache.width + 1] = hash_value
        row[cache.width + 2] = 1
    row[cache.width] = value


cdef class WideNumericCache:
    def __cinit__(self, Py_ssize_t width, bint private_owner=False, size_t capacity=8):
        if capacity < 8 or capacity & (capacity - 1):
            raise ValueError("numeric cache capacity must be a power of two of at least eight")
        if width < 4 or <size_t>width + 3 > (<size_t>-1) // sizeof(word) // capacity:
            raise ValueError("invalid wide numeric cache key width")
        self.width = width
        self.stride = self.width + 3
        self.private_owner = private_owner
        self.capacity = capacity
        self.entries = <word*>calloc(self.capacity, self.stride * sizeof(word))
        if self.entries == NULL:
            raise MemoryError()

    def __dealloc__(self):
        free(self.entries)

    cdef bint get(self, const word* key, bint reflexive, word hash_value, word* value) except? -1:
        if not reflexive:
            return False
        if self.private_owner:
            return read(self, key, hash_value, value)
        with cython.critical_section(self):
            return read(self, key, hash_value, value)

    cdef void put(self, const word* key, bint reflexive, word hash_value, word value) except *:
        if not reflexive:
            return
        if self.private_owner:
            write(self, key, hash_value, value)
            return
        with cython.critical_section(self):
            write(self, key, hash_value, value)

    cdef bint reserve(self, const word* key, bint reflexive, word hash_value,
                      Py_ssize_t* slot, word* value) noexcept:
        cdef word* row
        slot[0] = -1
        if not reflexive:
            return False
        row = position(self.entries, self.capacity, self.width, self.stride, key, hash_value)
        if row[self.width + 2]:
            value[0] = row[self.width]
            return True
        # Reserve no mutable state. The calculation may raise or recurse.
        slot[0] = (row - self.entries) // self.stride
        return False

    cdef void publish(self, const word* key, word hash_value, Py_ssize_t slot, word value) except *:
        cdef word* row
        if slot < 0:
            return
        # Reuse an empty home slot. Recursive growth and collision chains use
        # the normal insertion path, including its full key comparison.
        if self.used * 2 >= self.capacity or <size_t>slot != (hash_value & (self.capacity - 1)):
            write(self, key, hash_value, value)
            return
        row = self.entries + slot * self.stride
        if row[self.width + 2]:
            write(self, key, hash_value, value)
            return
        self.used += 1
        memcpy(row, key, self.width * sizeof(word))
        row[self.width] = value
        row[self.width + 1] = hash_value
        row[self.width + 2] = 1

    cpdef reset(self):
        with cython.critical_section(self):
            memset(self.entries, 0, self.capacity * self.stride * sizeof(word))
            self.used = 0

    def __reduce__(self):
        cdef size_t i, j
        cdef word* row
        values = []
        with cython.critical_section(self):
            for i in range(self.capacity):
                row = self.entries + i * self.stride
                if row[self.width + 2]:
                    key = []
                    for j in range(self.width):
                        key.append(row[j])
                    values.append((tuple(key), row[self.width]))
        return (WideNumericCache, (self.width, self.private_owner), values)

    def __setstate__(self, values):
        cdef size_t i
        cdef word* words = <word*>calloc(self.width, sizeof(word))
        if words == NULL:
            raise MemoryError()
        try:
            self.reset()
            for key, value in values:
                if <size_t>len(key) != self.width:
                    raise ValueError("invalid wide numeric cache key")
                for i in range(self.width):
                    words[i] = key[i]
                self.put(words, True, key_hash(words, self.width), value)
        finally:
            free(words)
