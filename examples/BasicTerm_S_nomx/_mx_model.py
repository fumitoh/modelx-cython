from . import _mx_sys
from . import _mx_classes

class _c_BasicTerm_S(_mx_sys.BaseModel):

    def __init__(self):

        # modelx variables
        self._parent = None
        self._model = self
        self._name = "BasicTerm_S"

        # Space assignments
        self.Projection = _mx_classes._c_Projection(self)
        self._mx_spaces = {
            'Projection': self.Projection
        }


        self._mx_load_io()

    def _mx_assign_refs(self, io_data, pickle_data):

        pass


mx_model = BasicTerm_S = _c_BasicTerm_S()
