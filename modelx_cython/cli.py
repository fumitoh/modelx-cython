# Copyright (c) 2023 Fumito Hamamura <fumito.ham@gmail.com>

# This library is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation version 3.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library.  If not, see <http://www.gnu.org/licenses/>.

import sys
import os
import textwrap
import pathlib
import shutil
import runpy
import ast
import argparse
import subprocess
from typing import IO, TYPE_CHECKING, Sequence, Optional, Tuple

from modelx_cython.consts import MX_MODEL_MOD, MX_SPACE_MOD, MX_SYS_MOD
from modelx_cython.tracer import trace_calls, MxCallTraceLogger, MxCodeFilter
from modelx_cython.config import TranslationSpec
from modelx_cython.transformer import SpaceTransformer, SpaceVisitor, PXDGenerator


def increment_backups(
        base_path: pathlib.Path,
        max_backups=1, nth=0):

    postfix = "_BAK" + str(nth) if nth else ""
    backup_path = pathlib.Path(str(base_path) + postfix)
    if backup_path.exists():
        if nth == max_backups:
            if backup_path.is_dir():
                shutil.rmtree(backup_path)
            elif backup_path.is_file():
                backup_path.unlink()
            else:
                raise ValueError("cannot remove '%s'" % str(backup_path))
        else:
            increment_backups(base_path, max_backups, nth + 1)
            next_backup = pathlib.Path(str(base_path) + "_BAK" + str(nth + 1))
            backup_path.rename(next_backup)


def run_sample(model_path: pathlib.Path, sample_path: str, new_model_name: str = None) -> MxCallTraceLogger:

    module: str = model_path.name
    try:
        module_path = str(model_path.parent)
        sys.path.insert(0, module_path)

        logger = MxCallTraceLogger(new_model_name=new_model_name)
        with trace_calls(
            module=module,
            logger=logger,
            max_typed_dict_size=0,
            code_filter= MxCodeFilter()):
            runpy.run_path(sample_path, run_name="__main__")

    finally:
        assert sys.path.pop(0) == module_path

    return logger


class HandlerError(Exception):
    pass


def main_handler(args: argparse.Namespace, stdout: IO[str], stderr: IO[str]) -> int:

    orig_path = pathlib.Path(args.model_path).resolve()
    work_dir = orig_path.parent
    model_name = orig_path.name + "_cy"
    model_path = work_dir / model_name
    setup_file = pathlib.Path(args.setup) if args.setup else work_dir / "setup.py"

    if not args.compile_only:
        increment_backups(model_path)
        shutil.copytree(orig_path, model_path)
        shutil.copy(pathlib.Path(__file__).parent / (MX_SYS_MOD + ".pxd"), model_path)

        logger = run_sample(orig_path, args.sample, new_model_name=model_name)
        spec = ast.literal_eval(pathlib.Path(args.spec).read_text())
        rel_model_path = model_path.relative_to(model_path.parent)

        modules = [rel_model_path / (MX_SYS_MOD + ".py")]
        for m in logger.modules:
            subs = m.split(".")
            assert subs.pop(0) == model_path.name
            assert subs[-1] in [MX_MODEL_MOD, MX_SPACE_MOD]
            pxd_path = subs.copy()
            subs[-1] = subs[-1] + ".py"
            pxd_path[-1] = pxd_path[-1] + ".pxd"
            abs_src_path = model_path / "/".join(subs)
            rel_src_path = rel_model_path / "/".join(subs)
            abs_pxd_path = model_path / "/".join(pxd_path)
            abs_init_path = model_path / "/".join(subs[:-1] + ["__init__.pxd"])

            visitor = SpaceVisitor(
                module_name=m,
                source=abs_src_path.read_text(),
                spec=TranslationSpec(spec),
                cells_info=logger.cells_info,
                ref_info=logger.ref_info,
            )

            trans = SpaceTransformer(visitor=visitor)
            pxd = PXDGenerator(visitor=visitor)

            abs_src_path.write_text(trans.transformed.code)
            abs_pxd_path.write_text(pxd.code)
            abs_init_path.write_text("from . cimport _mx_classes")
            modules.append(rel_src_path)

        create_setup(model_name, modules=modules, setup_file=setup_file)

    if args.translate_only:
        return 0
    else:
        return compile_main(work_dir, setup_file)


def compile_main(work_dir: pathlib.Path, setup_file: pathlib.Path) -> int:

    env = os.environ.copy()
    env["PYTHONPATH"] = str(work_dir) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = subprocess.run([sys.executable, str(setup_file), "build_ext", "--inplace"],
                         env=env, cwd=str(work_dir))
    return cmd.returncode


def main(argv: Sequence[str], stdout: IO[str], stderr: IO[str]) -> int:

    parser = argparse.ArgumentParser(
        description="Translate an exported modelx model into Cython and compile it.",
    )

    parser.add_argument(
        "model_path",
        type=str,
        help="Path to an exported modelx model to translate into Cython",
    )

    parser.add_argument(
        "--sample",
        type=str,
        default="sample.py",
        help=(
            "Path to a sample file to run for collecting type information (default: sample.py)"
        )
    )

    parser.add_argument(
        "--spec",
        type=str,
        default="spec.py",
        help=(
            "Path to a spec file for setting parameters (default: spec.py)"
        )
    )

    parser.add_argument(
        "--setup",
        type=str,
        default="",
        help=(
            "Path to a setup file for Cython (default: setup.py)"
        )
    )

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--translate-only",
        action="store_true",
        default=False,
        help="Perform translation only (default: False)",
    )

    group.add_argument(
        "--compile-only",
        action="store_true",
        default=False,
        help="Perform compilation only (default: False)",
    )


    args = parser.parse_args(argv)
    return main_handler(args, stdout, stderr)


def create_setup(model_name: str, modules: Sequence[str], setup_file: pathlib.Path):

    modules_str = textwrap.indent(",\n".join(
        ['"' + str(s) + '"' for s in modules]
    ), " " * 8)

    setup_script = textwrap.dedent("""\
    import sys
    from setuptools import setup
    from Cython.Build import cythonize

    setup(
        name="{model_name}",
        ext_modules=cythonize([
    {modules_str}
            ],
            annotate=True
        )
    )
    """)

    setup_file.write_text(
        setup_script.format(
            model_name=model_name,
            modules_str=modules_str))


def entry_point_main():
    """Wrapper for main() for setuptools console_script entry point."""
    # Since monkeytype needs to import the user's code (and possibly config
    # code), the user's code must be on the Python path. But when running the
    # CLI script, it won't be. So we add the current working directory to the
    # Python path ourselves.
    # sys.path.insert(0, os.getcwd())
    sys.exit(main(sys.argv[1:], sys.stdout, sys.stderr))
