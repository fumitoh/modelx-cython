"""Reuse native providers built for the same extension identity and toolchain."""

from contextlib import contextmanager
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import sysconfig
import tempfile
import time


def _file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _entry_lock(path):
    with path.open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            if not stream.tell():
                stream.write(b'\0')
                stream.flush()
            stream.seek(0)
            while True:
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as error:
                    if error.errno != errno.EACCES:
                        raise
                    time.sleep(0.1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX)
        yield  # Closing the file releases either platform lock.


def _compiler_identity(compiler):
    kind = compiler.compiler_type
    if kind == 'msvc':
        if not compiler.initialized:
            compiler.initialize()
        commands = [compiler.cc, compiler.linker]
        flags = [compiler.compile_options, compiler.compile_options_debug,
                 compiler.ldflags_shared, compiler.ldflags_shared_debug,
                 compiler.include_dirs, compiler.library_dirs]
    elif kind == 'unix':
        commands = [compiler.compiler_so, compiler.linker_so, compiler.compiler_cxx]
        flags = [compiler.include_dirs, compiler.library_dirs]
    else:
        return None
    binaries = []
    for command in commands:
        if not command:
            continue
        executable = command if isinstance(command, str) else command[0]
        executable = shutil.which(executable)
        if not executable or Path(executable).stem in {'ccache', 'sccache', 'distcc'}:
            return None
        binaries.append((executable, _file_hash(executable)))
    return kind, commands, flags, binaries


def provider_build_cache(providers):
    """Return a setuptools command for the explicitly configured build cache."""
    location = os.environ.get('MODELX_CYTHON_PROVIDER_CACHE')
    if not providers or not location:
        return {}

    from Cython import __version__ as cython_version
    from Cython.Build.Dependencies import create_dependency_tree
    from setuptools import __version__ as setuptools_version
    from setuptools.command.build_ext import build_ext

    root = Path(location).resolve()
    dependencies = create_dependency_tree()
    environment = {name: os.environ.get(name) for name in (
        'CC', 'CXX', 'LDSHARED', 'CFLAGS', 'CPPFLAGS', 'CXXFLAGS', 'LDFLAGS',
        'CL', '_CL_', 'LINK', '_LINK_', 'INCLUDE', 'LIB', 'LIBPATH',
        'SDKROOT', 'MACOSX_DEPLOYMENT_TARGET',
    )}
    platform = (sysconfig.get_platform(), sys.version, sysconfig.get_config_var('SOABI'),
                sysconfig.get_config_var('EXT_SUFFIX'), cython_version, setuptools_version)

    class CachedProviders(build_ext):
        def build_extensions(self):
            self._provider_toolchain = _compiler_identity(self.compiler)
            return super().build_extensions()

        def build_extension(self, extension):
            source = providers.get(extension.name)
            if source is None or self._provider_toolchain is None or self.force:
                return super().build_extension(extension)
            files = set(dependencies.all_dependencies(source)) | set(extension.sources) | set(extension.depends)
            if any(not Path(file).is_file() for file in files):
                return super().build_extension(extension)
            inputs = sorted((Path(file).name, _file_hash(file)) for file in files)
            options = {name: getattr(extension, name) for name in (
                'define_macros', 'undef_macros', 'include_dirs', 'libraries', 'library_dirs',
                'runtime_library_dirs', 'extra_objects', 'extra_compile_args', 'extra_link_args',
                'language', 'py_limited_api',
            )}
            options['extra_objects'] = [(file, _file_hash(file)) for file in options['extra_objects']]
            identity = (extension.name, platform, environment, self._provider_toolchain,
                        self.debug, options, inputs)
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            target = Path(self.get_ext_fullpath(extension.name))
            root.mkdir(parents=True, exist_ok=True)
            cache = root / f'{key}.bin'
            stamp = root / f'{key}.sha256'
            with _entry_lock(root / f'{key}.lock'):
                if (cache.is_file() and stamp.is_file()
                        and _file_hash(cache) == stamp.read_text(encoding='ascii')):
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(cache, target)
                    if self.verbose:
                        print(f'Reused native provider {extension.name}', flush=True)
                    return
                objects = self.compiler.object_filenames(extension.sources, output_dir=self.build_temp)
                if target.exists() or any(Path(obj).exists() for obj in objects):
                    # A standard incremental build can reuse objects from a
                    # different invocation. Publish only a fresh compilation.
                    return super().build_extension(extension)
                super().build_extension(extension)
                with tempfile.NamedTemporaryFile(dir=root, delete=False) as temporary:
                    scratch = Path(temporary.name)
                try:
                    shutil.copyfile(target, scratch)
                    os.replace(scratch, cache)
                    stamp.write_text(_file_hash(cache), encoding='ascii')
                finally:
                    if scratch.exists():
                        scratch.unlink()

    return {'build_ext': CachedProviders}
