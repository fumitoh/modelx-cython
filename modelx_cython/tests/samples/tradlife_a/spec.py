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
        }
    }
}
