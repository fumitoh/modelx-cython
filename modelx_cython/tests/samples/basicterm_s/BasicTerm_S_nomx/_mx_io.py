ios = (
{'disc_rate_ann.xlsx': {'type': 'PandasIO', 'file_type': 'excel'},
 'model_point_table.xlsx': {'type': 'PandasIO', 'file_type': 'excel'},
 'mort_table.xlsx': {'type': 'PandasIO', 'file_type': 'excel'}})

iospecs = (
{2327272187344: {'type': 'PandasData',
                 'io': 'disc_rate_ann.xlsx',
                 'kwargs': {'read_args': {'index_col': 0, 'engine': 'openpyxl'},
                            'squeeze': True,
                            'name': 'zero_spot',
                            'sheet': None}},
 2327277479632: {'type': 'PandasData',
                 'io': 'model_point_table.xlsx',
                 'kwargs': {'read_args': {'index_col': 0, 'engine': 'openpyxl'},
                            'squeeze': False,
                            'name': None,
                            'sheet': None}},
 2327272521616: {'type': 'PandasData',
                 'io': 'mort_table.xlsx',
                 'kwargs': {'read_args': {'index_col': 0, 'engine': 'openpyxl'},
                            'squeeze': False,
                            'name': None,
                            'sheet': None}}})
