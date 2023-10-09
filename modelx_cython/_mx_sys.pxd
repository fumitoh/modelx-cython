cdef class BaseMxObject:
    pass

cdef class BaseParent(BaseMxObject):

    cdef public dict  _mx_spaces
    cdef public BaseParent _parent
    cdef public BaseModel _model

cdef class BaseModel(BaseParent):
    pass

cdef class BaseSpace(BaseParent):
    
    cdef BaseSpace _space
    cdef dict _mx_itemspaces 
    cdef public list _mx_roots 

