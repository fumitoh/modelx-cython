{
    "spaces": {
        "CommTable": {
            # The actuarial functions take multiple int params with large
            # traced maxima (x, n, f up to ~115), so array caching would
            # allocate huge N-D arrays; store these cells in dicts instead.
            "cells": {
                "AnnDuenx": {"return_type": "object"},
                "AnnDuex": {"return_type": "object"},
                "Ax": {"return_type": "object"},
                "Axn": {"return_type": "object"},
                "Exn": {"return_type": "object"}
            }
        },
        "Projection": {
            # The sampled policies never reach maturity (their projections
            # end at the mortality-table limit), so pols_maturity is traced
            # as int although its maturity branch returns a float.
            "cells": {
                "pols_maturity": {"return_type": "float"}
            },
            "spaces": {
                "InnerProj": {
                    "cells": {
                        "pols_maturity": {"return_type": "float"}
                    }
                }
            }
        }
    }
}
