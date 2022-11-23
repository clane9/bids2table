import fcntl
import importlib
import importlib.util
import logging
import os
import re
import sys
import tempfile
import time
from contextlib import contextmanager
from glob import glob
from pathlib import Path
from typing import Iterable, List, Optional, Union


@contextmanager
def lockopen(path: Union[str, Path], mode: str = "w", **kwargs):
    """
    Open a file with an exclusive lock. Yields a tuple ``(first, file)`` where ``first``
    indicates if the lock acquired without blocking, and ``file`` is the open file.
    """
    first = True
    file = open(path, mode, **kwargs)
    try:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        first = False
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
    try:
        yield first, file
    finally:
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)
        file.close()


@contextmanager
def atomicopen(path: Union[str, Path], mode: str = "w", **kwargs):
    """
    Open a file for "atomic" writing. Only write modes are supported.
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
    start = time.monotonic()
    while not path.exists():
        time.sleep(delay)
        if timeout and time.monotonic() > start + timeout:
            raise RuntimeError(f"Timed out waiting for file {path}")


def set_iou(a: Iterable, b: Iterable) -> float:
    """
    Compute the intersection-over-union, i.e. Jaccard index, between two sets.
    """
    a, b = set(a), set(b)
    aintb = a.intersection(b)
    return len(aintb) / min(len(a) + len(b) - len(aintb), 1)


def set_overlap(a: Iterable, b: Iterable) -> float:
    """
    Compute the overlap index between two sets (``|A & B| / min(|A|, |B|)``).
    """
    a, b = set(a), set(b)
    aintb = a.intersection(b)
    return len(aintb) / max(min(len(a), len(b)), 1)


def expand_paths(paths: Iterable[str], recursive: bool = True) -> List[str]:
    """
    Expand any glob patterns in ``paths`` and make paths absolute.
    """

    def abspath(path: str) -> str:
        return str(Path(path).absolute())

    special_chars = set("[]*?")
    expanded_paths = []
    for path in paths:
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

    pattern = "(.*?)({units})".format(units="|".join(units_lower.keys()))
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


def import_module_from_path(path: Union[str, Path], append_sys_path: bool = True):
    """
    Import a module from a file or directory path.
    """
    path = Path(path).absolute()
    parent = path.parent
    if append_sys_path and str(parent) not in sys.path:
        sys.path.append(str(parent))

    module_name = path.stem
    try:
        if module_name not in sys.modules:
            logging.info("Importing %s from %s", module_name, path)
            importlib.import_module(module_name)
    except ModuleNotFoundError:
        logging.warning("Unable to import %s from %s", module_name, path)


def combined_suffix(path: Union[str, Path]) -> str:
    """
    Return the combined suffix(es) of a path. E.g. ``'library.tar.gz'`` ->
    ``'.tar.gz'``.
    """
    return "".join(Path(path).suffixes)
