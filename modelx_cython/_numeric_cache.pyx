# cython: freethreading_compatible=True
"""Exact fixed-signature numeric keys; no formula or policy execution state."""
from libc.stdlib cimport calloc, free
from libc.string cimport memset
cimport cython

cdef inline word mix(word value) noexcept:
    value ^= value >> 30
    value *= <word>0xbf58476d1ce4e5b9
    value ^= value >> 27
    value *= <word>0x94d049bb133111eb
    return value ^ (value >> 31)


cdef inline size_t start_slot(size_t capacity, word a, word b, word c) noexcept:
    return (mix(a) ^ mix(b + <word>0x9e3779b97f4a7c15)
            ^ mix(c + <word>0x517cc1b727220a95)) & (capacity - 1)


cdef inline size_t position(Entry* entries, size_t capacity, word a, word b, word c) noexcept:
    cdef size_t index = start_slot(capacity, a, b, c)
    while entries[index].ready:
        if entries[index].a == a and entries[index].b == b and entries[index].c == c:
            break
        index = (index + 1) & (capacity - 1)
    return index


cdef inline bint read(NumericCache cache, word a, word b, word c, word* value) noexcept:
    cdef size_t index = position(cache.entries, cache.capacity, a, b, c)
    if not cache.entries[index].ready:
        return False
    value[0] = cache.entries[index].value
    return True


cdef inline void write(NumericCache cache, word a, word b, word c, word value) except *:
    cdef Entry* grown
    cdef size_t index, i, capacity
    if cache.used * 2 >= cache.capacity:
        if cache.capacity > ((<size_t>-1) // sizeof(Entry)) // 2:
            raise MemoryError("numeric cache exceeds addressable storage")
        capacity = cache.capacity * 2
        grown = <Entry*>calloc(capacity, sizeof(Entry))
        if grown == NULL:
            raise MemoryError()
        for i in range(cache.capacity):
            if cache.entries[i].ready:
                index = position(grown, capacity, cache.entries[i].a,
                                 cache.entries[i].b, cache.entries[i].c)
                grown[index] = cache.entries[i]
        free(cache.entries)
        cache.entries = grown
        cache.capacity = capacity
    index = position(cache.entries, cache.capacity, a, b, c)
    if not cache.entries[index].ready:
        cache.used += 1
    cache.entries[index].a = a
    cache.entries[index].b = b
    cache.entries[index].c = c
    cache.entries[index].value = value
    cache.entries[index].ready = True


cdef class NumericCache:
    def __cinit__(self, size_t primary_capacity=0):
        self.capacity = 8
        self.entries = <Entry*>calloc(self.capacity, sizeof(Entry))
        if self.entries == NULL:
            raise MemoryError()
        if primary_capacity:
            # The compiler owns its resource budget. The storage representation
            # itself requires only a power of two with at least one free slot.
            if primary_capacity < 2 or primary_capacity & (primary_capacity - 1):
                raise ValueError("primary capacity must be a power of two of at least 2")
            if primary_capacity > (<size_t>-1) // sizeof(PrimaryEntry):
                raise MemoryError("primary cache exceeds addressable storage")
            self.primary = <PrimaryEntry*>calloc(primary_capacity, sizeof(PrimaryEntry))
            if self.primary == NULL:
                raise MemoryError()
            self.primary_capacity = primary_capacity

    def __dealloc__(self):
        free(self.primary)
        free(self.entries)

    def __reduce__(self):
        cdef size_t index
        values = []
        with cython.critical_section(self):
            for index in range(self.capacity):
                if self.entries[index].ready:
                    values.append((self.entries[index].a, self.entries[index].b,
                                   self.entries[index].c, self.entries[index].value))
            for index in range(self.primary_capacity):
                if self.primary[index].state == 2:
                    values.append((self.primary[index].a, self.primary[index].b,
                                   self.primary[index].c, self.primary[index].value))
        return (NumericCache, (self.primary_capacity,), values)

    def __setstate__(self, values):
        cdef word a, b, c, value, ignored
        cdef Py_ssize_t slot
        self.reset()
        for a, b, c, value in values:
            self.reserve(a, b, c, True, &slot, &ignored)
            self.publish(a, b, c, slot, value)

    cdef bint get(self, word a, word b, word c, bint reflexive, word* value) except? -1:
        if not reflexive:
            return False
        with cython.critical_section(self):
            return read(self, a, b, c, value)

    cdef void put(self, word a, word b, word c, bint reflexive, word value) except *:
        if reflexive:
            with cython.critical_section(self):
                write(self, a, b, c, value)

    cdef bint reserve(self, word a, word b, word c, bint reflexive, Py_ssize_t* slot,
                      word* value) noexcept:
        cdef size_t index
        cdef PrimaryEntry* entry
        if not reflexive:
            slot[0] = -2
            return False
        if self.primary_capacity:
            index = start_slot(self.primary_capacity, a, b, c)
            entry = self.primary + index
            while entry.state:
                if entry.a == a and entry.b == b and entry.c == c:
                    slot[0] = <Py_ssize_t>index
                    if entry.state == 2:
                        value[0] = entry.value
                        return True
                    return False
                index = (index + 1) & (self.primary_capacity - 1)
                entry = self.primary + index
            if self.primary_used < self.primary_capacity // 2:
                entry.a = a
                entry.b = b
                entry.c = c
                # An unready key reserves the probe position across reentry.
                # It is still a miss after an exception and remains retryable.
                entry.state = 1
                self.primary_used += 1
                slot[0] = <Py_ssize_t>index
                return False
        slot[0] = -1
        return read(self, a, b, c, value)

    cdef void publish(self, word a, word b, word c, Py_ssize_t slot, word value) except *:
        cdef PrimaryEntry* entry
        cdef word ignored
        if slot == -2:
            return
        if 0 <= slot < <Py_ssize_t>self.primary_capacity:
            entry = self.primary + slot
            if entry.state and entry.a == a and entry.b == b and entry.c == c:
                entry.value = value
                entry.state = 2
                return
        # Reset, replacement or an overflow miss may invalidate the hint.
        self.reserve(a, b, c, True, &slot, &ignored)
        if slot >= 0:
            self.primary[slot].value = value
            self.primary[slot].state = 2
        else:
            write(self, a, b, c, value)

    cpdef reset(self):
        with cython.critical_section(self):
            memset(self.entries, 0, self.capacity * sizeof(Entry))
            self.used = 0
            if self.primary_capacity:
                memset(self.primary, 0, self.primary_capacity * sizeof(PrimaryEntry))
                self.primary_used = 0
