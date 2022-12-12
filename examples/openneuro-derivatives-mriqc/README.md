# Organizing all OpenNeuro MRIQC results using bids2table

[MRIQC](https://github.com/nipreps/mriqc) outputs are available for a lot of the [OpenNeuro](https://openneuro.org/) datasets at [OpenNeuroDerivatives](https://github.com/OpenNeuroDerivatives).

In this example, we'll organize all the MRIQC anatomical and functional image quality metrics into a parquet database.

## Downloading datasets

To download the OpenNeuro MRIQC datasets, run

```sh
datalad install https://github.com/OpenNeuroDerivatives/OpenNeuroDerivatives.git
cd OpenNeuroDerivatives
for d in ds*-mriqc; do
  datalad install $d
done
```

By default, datalad does not fetch all the data files up front. To get all the json data we need, we can run

```sh
while read f; do
  if [[ ! -e $(readlink -f $f) ]]; then
    echo $f >> missing_files.txt
  fi
done < <(find ds*-mriqc -name '*.json')

cat missing_files.txt | xargs datalad get
```

## Generating list of subject directories

To generate a (sorted) list of all the subject directories to process, we can run

```sh
find OpenNeuroDerivatives/ds*-mriqc -name 'sub-*' \
  | grep '/sub-[0-9]\+$' | sort > paths_list.txt
```

This results in 5315 unique subject directories across 120 datasets (as of 2022/12/12).

## Configuring `bids2table`

We'll use a local config file [openneuro_mriqc.yaml](config/openneuro_mriqc.yaml) for this example.

The config itself does not contain many significant changes compared to the built-in [mriqc.yaml](../../bids2table/config/mriqc.yaml). *However*, our local config directory does contain modified BIDS indexer configs [bids_anat.yaml](config/tables/indexer/bids_anat.yaml) and [bids_func.yaml](config/tables/indexer/bids_func.yaml). By using a local version of the mriqc config, we can make sure our final config is composed with these changes.

### Overrides

We can also include a list of config *overrides* in a separate YAML file. Each entry in the list should be a `KEY: VALUE` pair. Nested config entries can be overridden by including the containing group(s), separated by dots, e.g. `GROUP.KEY: VALUE`.

Our overrides are in [overrides.yaml](overrides.yaml)

```yaml
- db_dir: OpenNeuroDerivatives/db
- log_dir: OpenNeuroDerivatives/bids2table.log
- paths.list_path: paths_list.txt
```

## Running bids2table

First we run `bids2table` with the `-p` option to check the composed config.

Note we also include a command-line override `collection_id=2022-12-12-1700`. Command line overrides work the same as YAML overrides. The `collection_id` is a mandatory identifier that is unique for each bids2table collection run.

A reasonable convention for the `collection_id` is to use the output of `date '+%Y-%m-%d-%H%M'`.

```sh
python -m bids2table -p -c config/openneuro_mriqc.yaml -y overrides.yaml \
    collection_id=2022-12-12-1730
```

Next we do a "dry run" that includes processing the first subject directory, but without saving any results.

```sh
python -m bids2table -c config/openneuro_mriqc.yaml -y overrides.yaml \
    collection_id=2022-12-12-1730 \
    dry_run=true
```

It seems the dry run succeeded

```
(0000) [INFO 22-12-12 17:32:58 engine.py: 160]: Finished crawl:
    path: /ocean/projects/med220004p/clane2/code/bids2table/examples/openneuro-derivatives-mriqc/OpenNeuroDerivatives/ds000001-mriqc/sub-01
    counts: {'total': 4, 'process': 4, 'error': 0}
    count totals: {'total': 4, 'process': 4, 'error': 0}
    runtime: 0.04 s throughput: 0 B/s
```

Now to run the full generation process, we use [SLURM](https://slurm.schedmd.com/) to run multiple workers in parallel. Specifically, we use a SLURM [job array](https://slurm.schedmd.com/job_array.html) and pass the `SLURM_ARRAY_TASK_ID` as an override for the `worker_id`. Internally, `bids2table` handles how to assign work to each worker.

```sh
python -m bids2table -c config/openneuro_mriqc.yaml -y overrides.yaml \
    collection_id=2022-12-12-1730 \
    worker_id=$SLURM_ARRAY_TASK_ID \
    num_workers=20
```

The whole process should run in about a minute.
