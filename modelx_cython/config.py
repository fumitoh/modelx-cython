
from modelx_cython.consts import (
    FILE_PREF,
    GLOBAL_PREF,
    MODULE_PREF,
    SPACE_PREF,
)


class TranslationSpec:

    SPACES = "spaces"
    SPACE_PARAMS = "space_params"
    CELLS = "cells"
    CELLS_PARAMS = "cells_params"
    SIZE = "size"
    RET_T = "return_type"
    PARAM_T = "param_type"


    def __init__(self, data: dict) -> None:
        
        self._data = data

    def get_spec(self, object_path: str):
        """
        Examples:

            PkgName._mx_classes._c_Projection
            PkgName._m_Parent._m_Child
        """
        def get_subspace(data, name):
            spaces = data.get(self.SPACES)
            if spaces and spaces.get(name):
                return spaces.get(name)
            else:
                return {}
            
        names = object_path.split(".")[1:]  # Remove first
        data = self._data
        while names:
            name = names.pop(0)
            if name[:len(FILE_PREF)] == FILE_PREF:
                continue
            elif name[:len(MODULE_PREF)] == MODULE_PREF:
                name = name[len(MODULE_PREF):]
                data_ = get_subspace(data, name)
                if data_:
                    data = data_
                else:
                    return {}
            elif name[:len(SPACE_PREF)] == SPACE_PREF:
                name = name[len(SPACE_PREF):]
                data_ = get_subspace(data, name)
                if data_:
                    data = data_
                else:
                    return {}

        return data
    
