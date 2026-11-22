"""Read native object declarations beside existing function-provider sources."""

from dataclasses import dataclass
from importlib import import_module
import json
import re
from types import MappingProxyType
from typing import Mapping, Optional

from .helpers import read_resource, valid_identifier


@dataclass(frozen=True)
class TypedHandle:
    """M1 typed handle of a provider type: the cdef class and the function that builds it."""
    type: str
    build: str


@dataclass(frozen=True)
class ProviderType:
    view_type: Optional[str]
    bind: Optional[str]
    admission: str
    bind_cell: bool
    handle: Optional[TypedHandle] = None


@dataclass(frozen=True)
class ProviderManifest:
    protocol: int
    provider: str
    abi: str
    types: Mapping[str, ProviderType]


@dataclass(frozen=True)
class ProviderBinding:
    manifest: ProviderManifest
    declaration: ProviderType
    type_id: str
    declared: bool

    @property
    def identity(self):
        return self.manifest.provider, self.manifest.abi, self.type_id


def observed_provider_binding(traces):
    """Read a registration only from the exact sampled owner's namespace."""
    owner = type(traces[0].ret_val)
    if vars(owner).get('__modelx_cython_type__') is None:
        return None
    identity = f'{owner.__module__}.{owner.__qualname__}'
    declared = True
    for trace in traces:
        annotation = trace.func.__annotations__.get('return')
        if (type(trace.ret_val) is not owner
                or annotation not in (None, owner, owner.__qualname__, identity)):
            return None
        declared &= annotation is not None
    return owner_provider_binding(owner, declared=declared)


def owner_provider_binding(owner, *, declared=False):
    """Resolve a provider for an already observed exact Python type."""
    registration = vars(owner).get('__modelx_cython_type__')
    if registration is None:
        return None
    if (not isinstance(registration, tuple) or len(registration) != 4
            or type(registration[0]) is not int or registration[0] != 1
            or not _qualified_identifier(registration[3])):
        identity = f'{owner.__module__}.{owner.__qualname__}'
        raise ValueError(f'invalid native type registration on {identity}')
    _, provider, abi, type_id = registration
    manifest = read_provider_manifest(provider, abi)
    if manifest is None:
        return None
    declaration = manifest.types.get(type_id)
    if declaration is None:
        raise ValueError(f'provider {provider!r} does not declare type {type_id!r}')
    return ProviderBinding(manifest, declaration, type_id, declared)


def _qualified_identifier(value):
    return (isinstance(value, str) and '.' in value
            and all(valid_identifier(part) for part in value.split('.')))


def _valid_abi(value):
    return isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value)


def _unique_fields(pairs):
    result = {}
    for field, value in pairs:
        if field in result:
            raise ValueError(f'duplicate field {field!r}')
        result[field] = value
    return result


def _require_fields(value, label, required, optional=()):
    if (not isinstance(value, dict) or set(required).difference(value)
            or set(value).difference(required, optional)):
        raise ValueError(f'invalid fields for {label}')


def _read_symbols(cls, value, label, required, optional=()):
    _require_fields(value, label, required, optional)
    if not all(valid_identifier(symbol) for symbol in value.values()):
        raise ValueError(f'invalid native symbol for {label}')
    return cls(**value)


def _read_type(type_id, declaration):
    if not _qualified_identifier(type_id):
        raise ValueError(f'invalid qualified type identity {type_id!r}')
    _require_fields(declaration, type_id, (),
                    ('view_type', 'bind', 'admission', 'bind_cell', 'handle'))
    view_type = declaration.get('view_type')
    bind = declaration.get('bind')
    if ('view_type' in declaration or 'bind' in declaration) and not (
        valid_identifier(view_type) and valid_identifier(bind)
    ):
        raise ValueError(f'{type_id} must declare valid view_type and bind symbols together')
    admission = declaration.get('admission', 'observed')
    bind_cell = declaration.get('bind_cell', False)
    if admission not in ('observed', 'declared_or_forwarded') or type(bind_cell) is not bool:
        raise ValueError(f'invalid view admission or bind_cell declaration for {type_id}')
    if view_type is None and (bind_cell or admission != 'observed'):
        raise ValueError(f'{type_id} cannot configure binding without a view_type')
    handle = None
    if 'handle' in declaration:
        handle = _read_symbols(TypedHandle, declaration['handle'], f'{type_id}.handle', ('type', 'build'))

    return ProviderType(view_type, bind, admission, bind_cell, handle)


def read_provider_manifest(provider: str, expected_abi: str) -> Optional[ProviderManifest]:
    """Read a provider's optional `<name>.provider.json` resource.

    Missing declarations permit the caller to retain its object path. A
    present malformed or incompatible declaration raises ValueError. The
    expected ABI comes from the semantic owner's registration. This reads
    data in the containing package without importing the provider module.
    Native signature support and the loaded binary are checked at binding.
    """
    if not _qualified_identifier(provider):
        raise ValueError(f'invalid packaged provider identity {provider!r}')
    if not _valid_abi(expected_abi):
        raise ValueError(f'invalid expected ABI identity for provider {provider!r}')
    package, _, name = provider.rpartition('.')
    try:
        owner_package = import_module(package)
    except ModuleNotFoundError as error:
        if error.name and (package == error.name or package.startswith(error.name + '.')):
            return None
        raise
    try:
        payload = read_resource(owner_package, name + '.provider.json')
    except FileNotFoundError:
        return None

    try:
        declaration = json.loads(payload, object_pairs_hook=_unique_fields)
        _require_fields(declaration, 'manifest', ('protocol', 'provider', 'abi', 'types'))
        protocol = declaration['protocol']
        if type(protocol) is not int or protocol != 1:
            raise ValueError(f'unsupported protocol {protocol!r}, expected 1')
        if declaration['provider'] != provider:
            raise ValueError(f'declares provider {declaration["provider"]!r}, expected {provider!r}')
        abi = declaration['abi']
        if not _valid_abi(abi):
            raise ValueError(f'invalid ABI identity {abi!r}')
        if abi != expected_abi:
            raise ValueError(f'declares ABI {abi!r}, expected {expected_abi!r}')
        types = declaration['types']
        if not isinstance(types, dict) or not types:
            raise ValueError('types must be a nonempty object')
        types = {type_id: _read_type(type_id, entry) for type_id, entry in types.items()}
    except ValueError as error:
        raise ValueError(f'invalid manifest for provider {provider!r}: {error}') from error

    return ProviderManifest(protocol, provider, abi, MappingProxyType(types))
