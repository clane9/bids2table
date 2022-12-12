import fcntl
import importlib
import importlib.util
import logging
import os
import re
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager
from fnmatch import fnmatch
from glob import glob
from pathlib import Path
from typing import (
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

T = TypeVar("T")


class PatternLUT(Generic[T]):
    """
    Lookup table for finding all matching patterns (and values) for a query string.

    Args:
        items: List of ``(pattern, value)`` tuples making up the lookup table. Only
            basic glob patterns are supported. The ``'*'`` wildcard matches any zero or
            more characters including the path separator. Patterns should use the posix
            path separator ``'/'``. Patterns without any path separators are matched
            only to the base file name.

    .. note::
        If matched files are expected to have a suffix, e.g. ".txt", the pattern must
        include the full suffix. Otherwise the matching will fail.
    """

    def __init__(self, items: List[Tuple[str, T]]):
        self.items = items

        # Organize items by suffix for faster lookup.
        self._items_by_suffix: Dict[str, List[Tuple[str, T]]] = defaultdict(list)
        for pattern, val in items:
            if pattern[-1] in "]*?":
                logging.warning(
                    f"Pattern '{pattern}' ends in a special character, assuming "
                    "empty suffix"
                )
                suffix = ""
            else:
                suffix = Path(pattern).suffix
            self._items_by_suffix[suffix].append((pattern, val))

    def lookup(self, path: Union[str, Path]) -> Iterator[Tuple[str, T]]:
        """
        Lookup one or more items for a path by glob pattern matching.
        """
        path = Path(path)
        for pattern, val in self._items_by_suffix[path.suffix]:
            # TODO: could consider generalizing this pattern matching to:
            #   - tuples of globs
            #   - arbitrary regex
            # But better to keep things simple for now.
            query = path.as_posix() if "/" in pattern else path.name
            if fnmatch(query, pattern):
                yield pattern, val


@contextmanager
def lockopen(path: Union[str, Path], mode: str = "w", **kwargs):
    """
    Open a file with an exclusive lock.

    See also: https://github.com/dmfrey/FileLock/blob/master/filelock/filelock.py
    """
    file = open(path, mode, **kwargs)
    fcntl.flock(file.fileno(), fcntl.LOCK_EX)
    try:
        yield file
    finally:
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)
        file.close()


@contextmanager
def atomicopen(path: Union[str, Path], mode: str = "w", **kwargs):
    """
    Open a file for "atomic" all-or-nothing writing. Only write modes are supported.
    """
    if mode[0] not in {"w", "x"}:
        raise ValueError(f"Only write modes supported; not '{mode}'")
    path = Path(path)
    file = tempfile.NamedTemporaryFile(
        mode=mode,
        dir=path.parent,
        prefix=".tmp-",
        suffix=path.suffix,
        delete=False,
        **kwargs,
    )
    try:
        yield file
    except Exception as exc:
        file.close()
        os.remove(file.name)
        raise exc
    else:
        file.close()
        os.replace(file.name, path)


@contextmanager
def waitopen(
    path: Union[str, Path],
    mode: str = "r",
    timeout: Optional[float] = None,
    delay: float = 0.1,
    **kwargs,
):
    """
    Wait for a file to exist before trying to open. Only read modes are supported.
    """
    if mode[0] != "r":
        raise ValueError(f"Only read modes supported; not '{mode}'")
    wait_for_file(path, timeout=timeout, delay=delay)
    file = open(path, mode=mode, **kwargs)
    try:
        yield file
    finally:
        file.close()


def wait_for_file(
    path: Union[str, Path],
    timeout: Optional[float] = None,
    delay: float = 0.1,
):
    """
    Wait for a file to exist.
    """
    path = Path(path)
    if timeout:
        delay = min(delay, timeout / 2)
    start = time.monotonic()
    while not path.exists():
        time.sleep(delay)
        if timeout and time.monotonic() > start + timeout:
            raise RuntimeError(f"Timed out waiting for file {path}")


def expand_paths(
    paths: Iterable[str],
    recursive: bool = True,
    root: Optional[Union[str, Path]] = None,
) -> List[str]:
    """
    Expand any glob patterns in ``paths`` and make paths absolute.
    """
    if root is not None:
        root = Path(root)

    def abspath(path: str) -> str:
        return str(Path(path).absolute())

    special_chars = set("[]*?")
    expanded_paths = []
    for path in paths:
        if root is not None:
            path = str(root / path)
        if special_chars.isdisjoint(path):
            expanded_paths.append(abspath(path))
        else:
            matched_paths = sorted(glob(path, recursive=recursive))
            expanded_paths.extend(map(abspath, matched_paths))
    return expanded_paths


def parse_size(size: str) -> int:
    """
    Parse a human readable size string like ``10MB`` to integer bytes.
    """
    units = {
        "B": 1,
        "KB": 10**3,
        "MB": 10**6,
        "GB": 10**9,
        "KiB": 1024,
        "MiB": 1024**2,
        "GiB": 1024**3,
    }
    units_lower = {k.lower(): v for k, v in units.items()}

    pattern = r"([0-9.\s]+)({units})".format(units="|".join(units_lower.keys()))
    match = re.match(pattern, size, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(
            f"Size {size} didn't match any of the following units:\n\t"
            + ", ".join(units.keys())
        )
    size = match.group(1)
    num = float(size)
    unit = match.group(2)
    bytesize = int(num * units_lower[unit.lower()])
    return bytesize


def detect_size_units(size: Union[int, float]) -> Tuple[float, str]:
    """
    Given ``size`` in bytes, find the best size unit and return ``size`` in those units.

    Example:
        >>> detect_size_units(2000)
        (2.0, 'KB')
    """
    if size < 1e3:
        return float(size), "B"
    elif size < 1e6:
        return size / 1e3, "KB"
    elif size < 1e9:
        return size / 1e6, "MB"
    else:
        return size / 1e9, "GB"


def import_module_from_path(path: Union[str, Path], prepend_sys_path: bool = True):
    """
    Import a module or package from a file or directory path.
    """
    path = Path(path).absolute()
    parent = path.parent
    module_name = path.stem
    logging.info("Importing %s from %s", module_name, path)
    with insert_sys_path(str(parent), prepend=prepend_sys_path):
        importlib.import_module(module_name)


@contextmanager
def insert_sys_path(path: str, prepend: bool = True):
    """
    Context manager to temporarily insert a path in ``sys.path``.
    """
    inserted = False
    if path not in sys.path:
        if prepend:
            sys.path.insert(0, path)
        else:
            sys.path.append(path)
        inserted = True
    try:
        yield sys.path
    finally:
        if inserted:
            sys.path.remove(path)
