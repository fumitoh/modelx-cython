# Copilot Instructions for modelx-cython

## Repository Overview

**Purpose**: Python package for cythonizing exported modelx models (actuarial/financial modeling). Translates pure-Python modelx models to Cython for performance.

**Type**: Python library with CLI tool | **Size**: ~84 files, ~3,800 LOC | **Status**: Alpha (experimental)

**Tech Stack**: Python 3.7+ (tested 3.12.3), Cython 3.0+, modelx 0.23+, libcst, MonkeyType, setuptools, numpy, pytest

## Development Setup & Build

**ALWAYS run in sequence:**
```bash
pip install -e .                    # Editable install, ~20s, installs ~8 deps
pip install pytest lifelib          # For tests
mx2cy --help                        # Verify
```

**Build:**
```bash
rm -rf build dist *.egg-info        # Clean first
python -m build                     # ~30-60s, creates dist/*.whl and *.tar.gz
```
Note: SetuptoolsDeprecationWarnings about licenses are non-fatal, ignore them.

## Testing

```bash
pytest                                                    # All tests (2-5 min)
pytest modelx_cython/tests/test_config.py -v             # Config tests (<1s)
pytest modelx_cython/tests/test_samples.py::test_no_spec -v  # Specific (5-15s)
```
**Requires**: `lifelib` for integration tests | **Structure**: Unit tests in `test_config.py`, integration in `test_samples.py` (15 scenarios in `tests/samples/`)

## Repository Structure

**Root**: `.github/workflows/` (PyPI CI), `modelx_cython/` (source), `examples/` (samples), `doc/source/` (minimal docs), `pyproject.toml`, `README.md`, `MANIFEST.in`

**Main Package** (`modelx_cython/`) - Data flow order:
1. `cli.py` - Entry point (`mx2cy` command) | 2. `tracer.py` - Runtime type collection (MonkeyType) | 3. `parser.py` - Parse models (libcst) | 4. `builder.py` - Combine parse/trace info | 5. `transformer.py` - Python→Cython transformation | 6. `config.py` - Config/spec handling | 7. `typedefs.py` - Type helpers | 8. `consts.py` - Constants/naming | 9. `monkeytype_tracing.py` - Custom tracer | 10. `_mx_sys.pxd` - Cython declarations | 11. `__init__.py` - Version (version_info tuple)

**Tests** (`modelx_cython/tests/`): `test_config.py` (unit), `test_samples.py` (integration, parameterized), `samples/` (15 scenarios: basicterm_s, basicterm_sc, array_size, deep_recursion, nested_params, ref_space, various_types, etc.)

## Workflow & Usage

**mx2cy Pipeline**: Export (user, outside tool) → Trace (run sample.py) → Parse (model structure) → Transform (Python→Cython) → Compile (setuptools)

**Commands**:
```bash
mx2cy Model_nomx --sample sample.py --spec spec.py  # Full pipeline (default)
mx2cy Model_nomx --sample sample.py --no-spec       # Without spec file
mx2cy Model_nomx --translate-only                    # Skip compilation
mx2cy Model_nomx --compile-only                      # Only compile (after translate)
```
**Key Files**: `sample.py` (exercises model), `spec.py` (optional config: array sizes, type overrides), `setup.py` (auto-generated)

**Making Changes**: Understand pipeline (tracer→parser→builder→transformer), test incrementally, check `examples/`. No linting config exists.

## CI/CD

**Workflows** (`.github/workflows/`):
1. `publish-to-pypi.yml` - On release tags: build→publish PyPI
2. `publish-to-testpypi.yml` - Manual: build→publish TestPyPI

**No automated test CI**. Tests run manually before release.

## Common Issues

**Import errors**: Run `pip install -e .` from repo root
**Missing deps**: Install `pip install pytest lifelib`
**"lifelib not found"**: `pip install lifelib`
**Tests hang**: Cython compilation takes 10-15s per test, normal
**"Stack not unlimited" warning**: modelx warning, ignore
**"spec.py not found"**: Use `--no-spec` flag or create spec.py
**Cython compile fails**: Need C compiler (gcc/clang/MSVC)

## Architecture

**Naming** (from `consts.py`): `_mx_*` (globals), `_m_*` (module/space), `_c_*` (class/space), `_f_*` (formula/func), `_v_*` (variable), `_has_*` (boolean flag)

**Key Concepts**: 
- Exported models: modelx→pure-Python packages
- Type collection: MonkeyType runtime tracing
- AST transform: libcst Python→Cython
- Parameterization: Array indexing for modelx spaces
- Caching: `@cached_property` used extensively

**Version**: `modelx_cython/__init__.py` → `version_info` tuple (currently 0.0.8)

## Constraints & Best Practices

**Constraints**: Python 3.7+ (uses cached_property), C compiler required, modelx models must be pre-exported, API unstable (alpha)

**Best Practices**:
1. ALWAYS use `pip install -e .` for dev
2. Run relevant tests before commits
3. Test with `examples/` directory
4. Ensure imports from `modelx_cython.*`
5. Remember: project is about type inference/annotation
6. Build before PR: `python -m build` must succeed
7. No linting enforced, follow existing style

**Quick Ref**: Install: `pip install -e .` | Test: `pytest` | Build: `python -m build` | CLI: `mx2cy --help` | Version: `modelx_cython/__init__.py` | Deps: `pyproject.toml`
