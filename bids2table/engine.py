import json
import logging
import os
import socket
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from bids2table import utils as ut
from bids2table.crawler import Crawler
from bids2table.env import collect_env_info
from bids2table.handlers import HANDLER_CATALOG, Handler
from bids2table.indexers import INDEXER_CATALOG, Indexer
from bids2table.logging import ProcessedLog, _setup_logging
from bids2table.writer import BufferedParquetWriter

IndexersMap = Dict[str, Indexer]
HandlersMap = Dict[str, List[Handler]]
WritersMap = Dict[str, BufferedParquetWriter]


@hydra.main(config_path="config", config_name="defaults")
def main(cfg: DictConfig):
    run_id: str = cfg.output.run_id
    task_id: int = cfg.workers.task_id
    log_dir = Path(cfg.output.log_dir)
    run_log_dir = log_dir / run_id
    if not cfg.dry_run:
        run_log_dir.mkdir(parents=True, exist_ok=True)

    _setup_logging(
        task_id,
        log_dir=None if cfg.dry_run else run_log_dir,
        level=cfg.log_level,
    )
    logging.info("Starting bids2table")
    logging.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    logging.info("Env:\n%s", collect_env_info())

    if cfg.include_modules:
        logging.info("Importing include modules")
        for path in cfg.include_modules:
            ut.import_module_from_path(path)

    logging.info("Initializing tables")
    indexers_map, handlers_map = _initialize_tables(cfg)
    logging.info("Loading paths")
    paths = _load_paths(cfg) if task_id == 0 else None

    if cfg.dry_run:
        logging.info("Completed dry run; exiting")
        return

    paths_path = run_log_dir / "paths_list.txt"
    if task_id == 0 and not paths_path.exists():
        logging.info(f"Saving expanded paths:\n\t{paths_path}")
        with ut.atomicopen(paths_path, "w") as f:
            np.savetxt(f, paths, fmt="%s")
    else:
        logging.info(f"Loading expanded paths:\n\t{paths_path}")
        with ut.waitopen(paths_path) as f:
            paths = np.loadtxt(f, dtype=str)

    workers_dir = run_log_dir / "workers"
    workers_dir.mkdir(exist_ok=True)
    worker_json = workers_dir / f"{_format_task_id(task_id)}.json"
    if worker_json.exists():
        logging.info(f"Task ID {task_id} was run previously; exiting")
        return
    hostname = socket.gethostname()
    pid = os.getpid()
    logging.info(f"Writing task ID: {task_id}\n\thostname: {hostname}\tpid: {pid}")
    with open(worker_json, "w") as f:
        json.dump({"task_id": task_id, "hostname": hostname, "pid": pid}, f)

    logging.info("Partitioning paths")
    paths = _partition_paths(cfg, paths)
    if len(paths) == 0:
        logging.info("No paths assigned to process; exiting")
        return

    logging.info("Starting main table generation process")
    _generate_tables(cfg, paths, indexers_map, handlers_map)
    return


def _generate_tables(
    cfg: DictConfig,
    paths: np.ndarray,
    indexers_map: IndexersMap,
    handlers_map: HandlersMap,
):
    """
    Main loop.

    Table partitions are written to::

        {db_dir} / {table_name} / {run_id} / {task_id} /
    """
    # TODO: does this need to depend on the cfg? have to balance whether this will be
    # used only in the cli context or more modularly. Maybe the crawler could be the
    # responsible module. In general, having every function here depend on the config,
    # often throughout the body, feels a little brittle.
    run_id: str = cfg.output.run_id
    task_id: int = cfg.workers.task_id
    task_id_str = _format_task_id(task_id)

    crawler = Crawler(handlers_map, indexers_map, **cfg.crawler)
    processed_log = ProcessedLog(cfg.output.db_dir)

    writers_map: WritersMap = {}
    for name in indexers_map:
        writer_dir = Path(cfg.output.db_dir) / name / run_id / task_id_str
        writer_dir.mkdir(parents=True)
        writers_map[name] = BufferedParquetWriter(prefix=str(writer_dir), **cfg.writer)
        logging.info(f"Initialized writer at directory:\n\t{writer_dir}")

    try:
        for path in paths:
            logging.info(f"Starting a directory crawl:\n\t{path}")
            # TODO: do something with errs
            tables, errs, counts = crawler(path)
            logging.info(
                f"Finished crawl:\n\tpath: {path}\n"
                f"\thandle count: {counts.count}\terrors: {counts.errors}\t"
                f"error_rate: {counts.error_rate:.3f}"
            )

            partitions = []
            for name, tab in tables.items():
                partition = writers_map[name].write(tab)
                partitions.append(partition)

            processed_log.write(
                run_id,
                task_id,
                path,
                counts.count,
                counts.errors,
                counts.error_rate,
                partitions,
            )

        # TODO: summary stats? throughput?

    finally:
        # TODO: might not want to flush last if exiting on exception.
        for name, writer in writers_map.items():
            writer.flush(blocking=True)


def _load_paths(cfg: DictConfig) -> np.ndarray:
    """
    Load the list of paths to process

    - Read the list from a file (.txt or .npy) or directly from the config
    - Expand any glob patterns
    - Optionally filter against the log of processed paths
    """
    paths_cfg = cfg.paths
    if isinstance(paths_cfg.list, str):
        logging.info("Loading paths from a file: %s", paths_cfg.list)
        list_path = Path(paths_cfg.list)
        if list_path.suffix == ".txt":
            paths = np.loadtxt(list_path, dtype=str)
        elif list_path.suffix == ".npy":
            paths = np.load(list_path)
        else:
            raise ValueError(
                "Expected the paths list file to be a '*.txt' or '*.npy'; "
                f"got {list_path.name}"
            )
    else:
        paths = paths_cfg.list
    expanded_paths = np.array(
        ut.expand_paths(paths, recursive=paths_cfg.glob_recursive)
    )
    logging.info("Loaded %d paths", len(expanded_paths))

    if paths_cfg.filter_completed:
        # TODO: I worry this part may be slow. Might be something best done offline. In
        # which case should expose this as a utility.
        processed_log = ProcessedLog(cfg.output.db_dir)
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


def _partition_paths(cfg: DictConfig, paths: np.ndarray) -> np.ndarray:
    """
    Partition the paths according to task worker and return the slice for this
    ``task_id``. Note the slice can be empty if ``cfg.paths.min_per_task`` is large.
    """
    task_id = cfg.workers.task_id
    num_tasks = cfg.workers.num_tasks
    task_num_paths = int(np.ceil(len(paths) / num_tasks))
    if cfg.paths.min_per_task:
        task_num_paths = max(task_num_paths, cfg.paths.min_per_task)
    start = task_id * task_num_paths
    if start >= len(paths):
        return np.array([], dtype=str)
    stop = min(len(paths), start + task_num_paths)
    return paths[start:stop]


def _initialize_tables(cfg: DictConfig) -> Tuple[IndexersMap, HandlersMap]:
    """
    For each table, initialize the indexer and handlers from the config.
    """
    indexers_map: IndexersMap = {}
    handlers_map: HandlersMap = defaultdict(list)

    # TODO: using a structured config could help with static and runtime type checking
    for table_cfg in cfg.tables:
        name = table_cfg.name
        logging.info("Initializing table %s", name)

        indexer_cfg: DictConfig = table_cfg.indexer.copy()
        indexer_name = indexer_cfg.pop("name")
        indexer_cls = INDEXER_CATALOG.get(indexer_name)
        if indexer_cls is None:
            raise ValueError(f"No indexer found matching '{indexer_name}'")
        indexer = indexer_cls.from_config(indexer_cfg)
        indexers_map[name] = indexer
        logging.info("Loaded indexer %s", indexer)

        for handler_name_or_pattern in table_cfg.handlers:
            handlers = HANDLER_CATALOG.search(handler_name_or_pattern)
            if len(handlers) == 0:
                raise ValueError(
                    f"No handlers found matching '{handler_name_or_pattern}'"
                )
            handlers_map[name].extend(handlers)
            logging.info(
                "Loaded handlers matching '%s':\n\t%s",
                handler_name_or_pattern,
                "\n\t".join(handler.name for handler in handlers),
            )
    return indexers_map, handlers_map


def _format_task_id(task_id: int) -> str:
    """
    Format a task ID as a zero-padded string.
    """
    return f"{task_id:05d}"
