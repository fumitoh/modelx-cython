import pathlib

# mx2cy must reject the names before it runs the sample
pathlib.Path(__file__).with_name("sample_ran.txt").write_text("ran")

from ReservedNames_nomx import mx_model

mx_model.Space1.total(1)
mx_model.Space1.check(1.0)
