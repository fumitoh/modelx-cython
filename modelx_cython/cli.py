import sys
import os
import pathlib
import shutil
import runpy
import ast
import argparse
from typing import IO, TYPE_CHECKING, List, Optional, Tuple

from modelx_cython.consts import MX_MODEL_MOD, MX_SPACE_MOD, MX_SYS_MOD
from modelx_cython.tracer import trace_calls, MxCallTraceLogger, MxCodeFilter
from modelx_cython.transformer import SpaceTransformer

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


def run_sample(model_path: pathlib.Path, new_model_name: str = None) -> MxCallTraceLogger:

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
            runpy.run_module("sample", run_name="__main__", alter_sys=True)

    finally:
        assert sys.path.pop(0) == module_path

    return logger


class HandlerError(Exception):
    pass


def translate_handler(args: argparse.Namespace, stdout: IO[str], stderr: IO[str]) -> None:

    orig_path = pathlib.Path(args.model_path).resolve()
    model_path = orig_path.parent / (orig_path.name + "_cy")
    increment_backups(model_path)
    shutil.copytree(orig_path, model_path)
    shutil.copy(pathlib.Path(__file__).parent / (MX_SYS_MOD + ".pxd"), model_path)

    logger = run_sample(orig_path, orig_path.name + "_cy")
    config = ast.literal_eval(pathlib.Path(args.paramfile).read_text())

    for m in logger.modules:
        subs = m.split(".")
        assert subs.pop(0) == model_path.name
        assert subs[-1] in [MX_MODEL_MOD, MX_SPACE_MOD]
        subs[-1] = subs[-1] + ".py"
        src_path = model_path / "/".join(subs)

        trans = SpaceTransformer(
            module_name=m, 
            source=src_path.read_text(),
            type_info=logger.type_info,
            ref_type_info=logger.ref_type_info,
            config=config["spaces"]
        )
        src_path.write_text(trans.transformed.code)


def main(argv: List[str], stdout: IO[str], stderr: IO[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Generate and apply stub files from collected type information.",
    )

    subparsers = parser.add_subparsers(title="commands", dest="command")

    translate_parser = subparsers.add_parser(
        "translate",
        help="Convert an exported modelx model into Cython sources",
        description="Convert an exported modelx model into Cython sources",
    )
    translate_parser.add_argument(
        "model_path",
        type=str,
        help="Path to an exported modelx model to translate into Cython",
    )
    translate_parser.add_argument(
        "--paramfile",
        type=str,
        default="parameters.py",
        help=(
            "Path to a parameter file for setting translation parameters"
        )
    )
    translate_parser.set_defaults(handler=translate_handler)


    args = parser.parse_args(argv)

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help(file=stderr)
        return 1

    try:
        handler(args, stdout, stderr)
    except HandlerError as err:
        print(f"ERROR: {err}", file=stderr)
        return 1

    return 0


def entry_point_main():
    """Wrapper for main() for setuptools console_script entry point."""
    # Since monkeytype needs to import the user's code (and possibly config
    # code), the user's code must be on the Python path. But when running the
    # CLI script, it won't be. So we add the current working directory to the
    # Python path ourselves.
    # sys.path.insert(0, os.getcwd())
    sys.exit(main(sys.argv[1:], sys.stdout, sys.stderr))
