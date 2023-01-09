"""
Main bids2table ETL engine.

- **Input**: Structured directory of data files (e.g. a BIDS directory).
- **Output**: Parquet database.

Example:

.. code-block:: python

    from hydra import compose, initialize_config_module
    from bids2table import launch

    overrides = ["collection_id=20221212", "dry_run=true"]
    with initialize_config_module("bids2table.config", job_name="bids2table"):
        cfg = compose(config_name="mriqc", overrides=overrides)
    launch(cfg)
"""

import logging
import os
import socket
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from omegaconf import OmegaConf

from bids2table import utils as ut
from bids2table.config import Config
from bids2table.crawler import CrawlCounts, Crawler
from bids2table.env import collect_env_info
from bids2table.handlers import HandlerTuple, get_handler
from bids2table.indexers import Indexer, get_indexer
from bids2table.logging import ProcessedLog, format_worker_id, setup_logging
from bids2table.writer import BufferedParquetWriter

IndexersMap = Dict[str, Indexer]
HandlersMap = Dict[str, List[HandlerTuple]]
WritersMap = Dict[str, BufferedParquetWriter]


def launch(cfg: Config):
    """
    Launch the bids2table collection task defined by the config ``cfg``, which should be
    compatible with the `Hydra structured config`_ :class:`bids2table.config.Config`.

    The primary input is a list of data directories defined in ``cfg.paths``.

    The output is one Parquet directory per table defined in ``cfg.tables``. Parquet
    directories are organized as::

        {db_dir} / {table_name} / {collection_id} / {worker_id} / {partition}.parquet

    .. _Hydra structured config: https://hydra.cc/docs/tutorials/structured_config/intro/
    """
    run_log_dir = Path(cfg.log_dir) / cfg.collection_id
    if not cfg.dry_run:
        run_log_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(
        cfg.worker_id,
        log_dir=(None if cfg.dry_run else run_log_dir),
        level=cfg.log_level,
    )
    logging.info("Starting bids2table")
    logging.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    logging.info("Env:\n%s", collect_env_info())

    missing_keys = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise ValueError(f"Missing keys in config: {missing_keys}")

    if cfg.dry_run and cfg.worker_id != 0:
        logging.info("Dry run only supported in head worker (worker_id=0); exiting")
        return

    if cfg.include_modules:
        logging.info("Importing include modules")
        for path in cfg.include_modules:
            ut.import_module_from_path(path)

    logging.info("Initializing tables")
    indexers_map, handlers_map = _initialize_tables(cfg)
    logging.info("Loading paths")
    paths = _load_paths(cfg) if cfg.worker_id == 0 else None

    paths_path = run_log_dir / "paths_list.txt"
    if cfg.worker_id == 0 and not paths_path.exists():
        logging.info(f"Saving expanded paths:\n\t{paths_path}")
        if not cfg.dry_run:
            with ut.atomicopen(paths_path, "w") as f:
                np.savetxt(f, paths, fmt="%s")
    else:
        logging.info(f"Loading expanded paths:\n\t{paths_path}")
        with ut.waitopen(paths_path) as f:
            paths = np.loadtxt(f, dtype=str)

    workers_dir = run_log_dir / "workers"
    if not cfg.dry_run:
        workers_dir.mkdir(exist_ok=True)

    worker_json = workers_dir / f"{format_worker_id(cfg.worker_id)}.json"
    if worker_json.exists():
        logging.info(f"Worker ID {cfg.worker_id} was run previously; exiting")
        return
    hostname = socket.gethostname()
    pid = os.getpid()
    logging.info(
        f"Writing worker ID: {cfg.worker_id}\n\thostname: {hostname}\tpid: {pid}"
    )
    if not cfg.dry_run:
        with open(worker_json, "w") as f:
            worker_dict = {"id": cfg.worker_id, "hostname": hostname, "pid": pid}
            print(worker_dict, file=f)

    logging.info("Partitioning paths")
    paths = _partition_paths(cfg, paths)
    if len(paths) == 0:
        logging.info("No paths assigned to process; exiting")
        return

    logging.info("Starting main table generation process")
    _generate_tables(cfg, paths, indexers_map, handlers_map)


def _generate_tables(
    cfg: Config,
    paths: Iterable[str],
    indexers_map: IndexersMap,
    handlers_map: HandlersMap,
):
    """
    Main loop.
    """
    worker_id_str = format_worker_id(cfg.worker_id)

    crawler = Crawler(indexers_map, handlers_map, **(cfg.crawler or {}))
    processed_log = ProcessedLog(cfg.db_dir)

    writers_map: WritersMap = {}
    for name in indexers_map:
        writer_dir = Path(cfg.db_dir) / name / cfg.collection_id / worker_id_str
        writers_map[name] = BufferedParquetWriter(
            prefix=str(writer_dir), **(cfg.writer or {})
        )
        logging.info(f"Initialized writer at directory:\n\t{writer_dir}")

    def _flush():
        if not cfg.dry_run:
            for _, writer in writers_map.items():
                writer.flush(blocking=True)

    tic = time.monotonic()
    dir_counts = CrawlCounts()
    count_totals = CrawlCounts()

    def progress_stats():
        rtime = time.monotonic() - tic
        total_bytes = sum(w.total_bytes() for w in writers_map.values())
        tput, tput_units = ut.detect_size_units(total_bytes / rtime)
        return rtime, tput, tput_units

    try:
        for path in paths:
            dir_counts.total += 1
            path = str(path)
            if not Path(path).is_dir():
                logging.warning(f"Skipping path {path}; not a directory")
                continue

            # TODO: should we be failing on first crawler exception?
            tables, errs, counts = crawler.crawl(path)
            count_totals.update(counts)
            dir_counts.process += 1

            if (
                logging.INFO >= logging.root.level
                and dir_counts.process % cfg.log_frequency == 0
            ):
                rtime, tput, tput_units = progress_stats()
                logging.info(
                    f"Crawler progress:\n"
                    f"\tlast dir: {path}\n"
                    f"\tlast dir file counts: {counts.to_dict()}\n"
                    f"\tdir counts: {dir_counts.to_dict()}\n"
                    f"\ttotal file counts: {count_totals.to_dict()}\n"
                    f"\truntime: {rtime:.2f} s\tthroughput: {tput:.0f} {tput_units}/s"
                )

            if cfg.dry_run:
                for name, tab in tables.items():
                    logging.info("Table: %s\n%s", name, tab)
                break

            partitions = []
            for name, tab in tables.items():
                partition = writers_map[name].write(tab)
                partitions.append(partition)

            processed_log.write(
                collection_id=cfg.collection_id,
                worker_id=cfg.worker_id,
                path=path,
                counts=counts,
                partitions=partitions,
                errors=errs,
            )

    except Exception:
        if cfg.flush_on_error:
            _flush()
        raise
    else:
        _flush()
    finally:
        crawler.close()
        rtime, tput, tput_units = progress_stats()
        logging.info(
            f"Crawler done:\n"
            f"\tdir counts: {dir_counts.to_dict()}\n"
            f"\ttotal file counts: {count_totals.to_dict()}\n"
            f"\truntime: {rtime:.2f} s\tthroughput: {tput:.0f} {tput_units}/s"
        )


def _load_paths(cfg: Config) -> np.ndarray:
    """
    Load the list of paths to process

    - Read the list from a file (.txt or .npy) or directly from the config
    - Expand any glob patterns
    - Optionally filter against the log of processed paths
    """
    paths_cfg = cfg.paths
    if paths_cfg.list_path:
        logging.info("Loading paths from a file: %s", paths_cfg.list_path)
        list_path = Path(paths_cfg.list_path)
        if list_path.suffix == ".txt":
            paths = np.loadtxt(list_path, dtype=str)
        elif list_path.suffix == ".npy":
            paths = np.load(list_path)
        else:
            raise ValueError(
                "Expected the paths list file to be a '*.txt' or '*.npy'; "
                f"got {list_path.name}"
            )
    elif paths_cfg.list:
        paths = paths_cfg.list
    else:
        raise ValueError("cfg.paths.list_path or cfg.paths.list is required")

    expanded_paths = np.array(
        ut.expand_paths(paths, recursive=paths_cfg.glob_recursive)
    )
    logging.info("Loaded %d paths", len(expanded_paths))

    if paths_cfg.filter_completed:
        # TODO: I worry this part may be slow. Might be something best done offline. In
        # which case should expose this as a utility.
        processed_log = ProcessedLog(cfg.db_dir)
        filtered_paths = processed_log.filter_paths(
            expanded_paths, error_rate_threshold=paths_cfg.redo_error_rate_threshold
        )
        logging.info(
            "Filtered out %d paths that were previously processed successfully",
            len(expanded_paths) - len(filtered_paths),
        )
    else:
        filtered_paths = expanded_paths

    if len(filtered_paths) > 10:
        paths_summary = (
            filtered_paths[:5].tolist() + ["..."] + filtered_paths[-5:].tolist()
        )
    else:
        paths_summary = filtered_paths.tolist()
    paths_summary = "\n\t".join(paths_summary)
    logging.info("Processing %d paths:\n\t%s", len(filtered_paths), paths_summary)
    return filtered_paths


def _partition_paths(cfg: Config, paths: np.ndarray) -> np.ndarray:
    """
    Partition the paths according to worker and return the slice for this ``worker_id``.
    Note the slice can be empty if ``cfg.paths.min_per_worker`` is large.
    """
    worker_id = cfg.worker_id
    num_workers = cfg.num_workers
    worker_num_paths = int(np.ceil(len(paths) / num_workers))
    if cfg.paths.min_per_worker:
        worker_num_paths = max(worker_num_paths, cfg.paths.min_per_worker)
    start = worker_id * worker_num_paths
    if start >= len(paths):
        return np.array([], dtype=str)
    stop = min(len(paths), start + worker_num_paths)
    sub_paths = paths[start:stop]
    return sub_paths


def _initialize_tables(cfg: Config) -> Tuple[IndexersMap, HandlersMap]:
    """
    For each table, initialize the indexer and handlers from the config.
    """
    indexers_map: IndexersMap = {}
    handlers_map: HandlersMap = defaultdict(list)

    for name, table_cfg in cfg.tables.items():
        if table_cfg.name:
            name = table_cfg.name
        logging.info("Initializing table %s", name)

        indexer_cfg = table_cfg.indexer
        indexer_cls = get_indexer(indexer_cfg.name)
        indexer = indexer_cls.from_config(indexer_cfg)
        indexers_map[name] = indexer
        logging.info(f"Loaded indexer: {indexer}")

        for handler_cfg in table_cfg.handlers.values():
            handler_cls = get_handler(handler_cfg.name)
            handler = handler_cls.from_config(handler_cfg)
            handlers_map[name].append(
                HandlerTuple(name, handler_cfg.pattern, handler_cfg.label, handler)
            )
            logging.info(
                f"Loaded handler: {handler}\n"
                f"\tpattern: {handler_cfg.pattern}\n"
                f"\tlabel: {handler_cfg.label}"
            )
    return indexers_map, handlers_map
