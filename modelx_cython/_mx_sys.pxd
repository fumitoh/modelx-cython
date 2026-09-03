cdef extern from *:
    """
    /* Acquire/release accessors for the _has_ flags of locked Spaces.
       The writer stores the cached value, then the flag with release
       semantics, under the model lock; a reader that sees the flag with
       acquire semantics is therefore guaranteed to see the value, without
       taking the lock. pyatomic.h is included by Python.h from CPython 3.13,
       the first version with a free-threaded build; a build with the GIL
       needs no ordering, and neither does an older version of Python. */
    #if defined(Py_GIL_DISABLED)
    #define __MX_LOAD_FLAG(p)      _Py_atomic_load_int_acquire((const int *)(p))
    #define __MX_STORE_FLAG(p, v)  _Py_atomic_store_int_release((int *)(p), (v))
    #else
    #define __MX_LOAD_FLAG(p)      (*(p))
    #define __MX_STORE_FLAG(p, v)  ((void)(*(p) = (v)))
    #endif
    """
    bint _mx_load_flag "__MX_LOAD_FLAG" (bint *p) noexcept nogil
    void _mx_store_flag "__MX_STORE_FLAG" (bint *p, bint v) noexcept nogil

cdef class BaseMxObject:
    pass

cdef class BaseParent(BaseMxObject):

    cdef public dict  _mx_spaces
    cdef public BaseParent _parent
    cdef public BaseModel _model
    cdef public str _name
    # The threading.RLock shared by the locked Spaces of the model, or None.
    # Declared on the common base of the model and of the Spaces: a locked
    # Space reads it through its typed _model field and through itself.
    cdef public object _mx_lock

cdef class BaseModel(BaseParent):
    pass

cdef class BaseSpace(BaseParent):

    cdef bint _mx_is_cells_set
    cdef dict _mx_cells

    cdef BaseSpace _space
    cdef dict _mx_itemspaces
    cdef public list _mx_roots
