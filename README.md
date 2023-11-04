# modelx-cython: Cythonize modelx models

**Important:**
modelx-cython is currently highly experimental and is released only for early adopters with advanced software engineering skills to explore its capability.

[modelx]:(https://github.com/fumitoh/modelx)

## What is modelx-cython?

modelx-cython is a Python package for cythonizing exported [modelx] models.

![modelx Ecosystem](https://raw.githubusercontent.com/fumitoh/modelx-cython/main/doc/source/images/modelx-ecosystem.png)

## Usage

Suppose `model` in the code below is a modelx Model.
The code creates a pure-Python model of `model` as a Python package named "Model_nomx" in the current directory.

```python
>>> model.export("Model_nomx")
```

You need a config file and a sample file to translate the model to a Cythonized model and compile it. 
Create and name them `config.py` and `sample.py` respectively, and place them in the directory where *Model_nomx* is located.

Upon installing modelx-cython, the `mx2cy` command becomes available as an executable command.
In a Python-enabled command prompt or shell, change the current directory with `cd` to where *Model_nomx* is located, and run:

```
mx2cy Model_nomx
```

This will create a Python package named "Model_nomx_cy", next to the original "Model_nomx".

## Command

```
usage: mx2cy [-h] [--sample SAMPLE] [--spec SPEC] [--setup SETUP] [--translate-only | --compile-only] model_path

Translate an exported modelx model into Cython and compile it.

positional arguments:
  model_path        Path to an exported modelx model to translate into Cython

options:
  -h, --help        show this help message and exit
  --sample SAMPLE   Path to a sample file to run for collecting type information (default: sample.py)
  --spec SPEC       Path to a spec file for setting parameters (default: spec.py)
  --setup SETUP     Path to a setup file for Cython (default: setup.py)
  --translate-only  Perform translation only (default: False)
  --compile-only    Perform compilation only (default: False)
```

## See Also

* [modelx GitHub Repository](https://github.com/fumitoh/modelx)
* [modelx Documentation](https://docs.modelx.io)
* [modelx Blog](https://modelx.io/allposts)

## How to Install

To install modelx-cython, run the following command from your command prompt:

```
pip install modelx-cython
```

If you're on Anaconda, instead above install it from conda-forge:

```
conda install -c conda-forge modelx-cython
```

## Requirements

modelx-cython requires Python 3.7+.

- modelx v0.23.0+
- Cython v3.0.0+
- setuptools
- libcst
- MonkeyType

## License

Copyright 2023, Fumito Hamamura

modelx is free software; you can redistribute it and/or
modify it under the terms of
[GNU Lesser General Public License v3 (LGPLv3)](https://github.com/fumitoh/modelx-cython/blob/master/LICENSE.txt).

