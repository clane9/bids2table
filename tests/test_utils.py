import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import pytest

from bids2table import utils as ut

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def pattern_lut() -> ut.PatternLUT:
    patterns = ["*.txt", "*_blah.txt", "blah*", "foo/*.gz"]
    items = [(p, ii) for ii, p in enumerate(patterns)]
    lut = ut.PatternLUT(items)
    return lut


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("a.txt", [0]),
        ("blah/a.txt", [0]),
        ("blah.txt", [0]),
        ("a_blah.txt", [0, 1]),
        ("blah_blah.txt", [0, 1]),
        ("blah", [2]),
        ("blah_blah", [2]),
        ("foo/a.gz", [3]),
        ("foo/a/b.gz", [3]),  # TODO: might not want this
        ("blah/foo/a.gz", []),  # TODO: might not want this
        ("bar", []),
    ],
)
def test_pattern_lut(pattern_lut: ut.PatternLUT, path: str, expected: List[int]):
    matches = list(pattern_lut.lookup(path))
    logging.debug(f"path={path}\tmatches={matches}")
    value_matches = [v[1] for v in matches]
    assert value_matches == expected


def test_lockopen(tmp_path: Path):
    fname = tmp_path / "file.txt"

    def _task(task_id: int):
        with ut.lockopen(fname, mode="a+") as f:
            t = time.time()
            lines = f.readlines()
            last_line = lines[-1] if lines else ""
            new_line = f"id={task_id}, t={t}"
            logging.debug(f"last line={last_line}\tnew line={new_line}")
            time.sleep(0.01)
            print(new_line, file=f)
        return task_id

    num_tasks = 10
    pool = ThreadPoolExecutor(4)
    for task_id in pool.map(_task, range(num_tasks)):
        logging.debug(f"Done {task_id}")

    with open(fname) as f:
        lines = f.readlines()
    assert len(lines) == num_tasks


def test_atomicopen(tmp_path: Path):
    fname = tmp_path / "file.txt"
    with ut.atomicopen(fname, "x") as f:
        logging.info(f"atomic temp file: {f.name}")
        print("blah", file=f)
        assert not fname.exists()
    assert fname.exists()
    assert not Path(f.name).exists()


def test_waitopen(tmp_path: Path):
    fname = tmp_path / "file.txt"

    def _delayed_create():
        time.sleep(0.1)
        with open(fname, "w") as f:
            print("", file=f)

    pool = ThreadPoolExecutor(1)
    pool.submit(_delayed_create)

    # too short of a wait
    with pytest.raises(RuntimeError):
        with ut.waitopen(fname, timeout=0.01):
            pass

    tic = time.monotonic()
    with ut.waitopen(fname, timeout=1.0):
        delay = time.monotonic() - tic
        logging.debug(f"opened file after waiting for {delay:.4f}s")


def test_waitopen_mode():
    with pytest.raises(ValueError):
        with ut.waitopen("file.txt", "w"):
            pass


def test_atomicopen_error(tmp_path: Path):
    fname = tmp_path / "file.txt"
    try:
        with ut.atomicopen(fname, "x") as f:
            raise RuntimeError
    except RuntimeError:
        pass
    assert not fname.exists()
    assert not Path(f.name).exists()


def test_atomicopen_mode():
    with pytest.raises(ValueError):
        with ut.atomicopen("file.txt", "r"):
            pass


def test_expand_paths():
    patterns = ["sub-*", "sub_bad", "**/*_T1w.json"]
    root = DATA_DIR / "ds000102-mriqc"
    expanded_paths = ut.expand_paths(patterns, recursive=True, root=root)
    expanded_paths = [str(Path(p).relative_to(root)) for p in expanded_paths]
    logging.info(f"path patterns={patterns}\nexpanded paths={expanded_paths}")
    assert len(expanded_paths) == 10


@pytest.mark.parametrize(
    ("alias", "expected"),
    [("1GB", int(1e9)), ("1  gb", int(1e9)), ("7.23 KiB", 7403), (" 1 KB ", 1000)],
)
def test_parse_size(alias: str, expected: int):
    size = ut.parse_size(alias)
    assert size == expected


def test_parse_size_error():
    with pytest.raises(ValueError):
        ut.parse_size("10 PB")


@pytest.fixture
def dummy_module(tmp_path: Path) -> Path:
    mod_path = tmp_path / "dummy_module.py"
    with open(mod_path, "w") as f:
        print("dummy = True", file=f)
    return mod_path


@pytest.fixture
def dummy_package(tmp_path: Path) -> Path:
    pkg_path = tmp_path / "dummy_package"
    os.mkdir(pkg_path)
    with open(pkg_path / "__init__.py", "w") as f:
        print("dummy = True", file=f)
        print("from . import blah", file=f)
    with open(pkg_path / "blah.py", "w") as f:
        print("blah = 12", file=f)
    return pkg_path


def test_insert_sys_path_prepend():
    path = "/dummy"
    with ut.insert_sys_path(path, prepend=True):
        assert sys.path[0] == path
    assert path not in sys.path


def test_insert_sys_path_append():
    path = "/dummy"
    with ut.insert_sys_path(path, prepend=False):
        assert sys.path[-1] == path
    assert path not in sys.path


def test_insert_sys_path_pre_exist():
    """
    Check that if a path is already present, nothing happens.
    """
    path = "/dummy"
    sys.path.append(path)
    with ut.insert_sys_path(path, prepend=True):
        assert sys.path[0] != path
    assert path in sys.path


def test_import_module_from_path(dummy_module: Path):
    ut.import_module_from_path(dummy_module)
    mod = sys.modules.get(dummy_module.stem)
    assert mod is not None
    assert hasattr(mod, "dummy") and mod.dummy


def test_import_module_from_path_pkg(dummy_package: Path):
    ut.import_module_from_path(dummy_package)
    pkg = sys.modules.get(dummy_package.stem)
    assert pkg is not None
    mod = getattr(pkg, "blah")
    assert mod is not None
    assert hasattr(mod, "blah") and mod.blah == 12


def test_import_module_from_path_fail(tmp_path: Path):
    with pytest.raises(ModuleNotFoundError):
        ut.import_module_from_path(tmp_path / "nonexist.py")


if __name__ == "__main__":
    pytest.main([__file__])
