from . import _mx_sys
from . import _mx_classes

class _c_BasicTerm_SC(_mx_sys.BaseModel):

    def __init__(self):

        # modelx variables
        self._parent = None
        self._model = self
        self._name = "BasicTerm_SC"

        # Space assignments
        self.Projection = _mx_classes._c_Projection(self)
        self.Data = _mx_classes._c_Data(self)
        self._mx_spaces = {
            'Projection': self.Projection,
            'Data': self.Data
        }


        self._mx_load_io()

    def _mx_assign_refs(self, io_data, pickle_data):

        pass


mx_model = BasicTerm_SC = _c_BasicTerm_SC()
