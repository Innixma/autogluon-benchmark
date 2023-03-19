Contents of each metadata file:

### task_metadata.csv

Contains the AutoMLBenchmark 104 datasets (study 271 and 269)

Generated from `autogluon_benchmark.metadata.metadata_generator.generate_metadata(study=[271, 269])`

### task_metadata_289.csv

Contains the AutoMLBenchmark 104 datasets (study 271 and 269), plus the 208 datasets from study 293.

These 312 datasets were then deduped by `did` resulting in 289 datasets.

Then, all tasks were standardized to `10-fold Crossvalidation` estimation_procedure (`estimation_procedure_id=1`)

Note that these 289 datasets might need further cleaning, as some datasets from study 293 appear to be corrupted.

### task_metadata_old.csv

Contains the AutoMLBenchmark 104 datasets (study 271 and 269) prior to dataset ID fixes/updates

4 datasets have a different `tid` than they do in `task_metadata.csv`, but are the same tasks in practice.

This is needed to fix any `tid` mismatches from old runs of AMLB before the changes.

Differences occur in the following 4 datasets with `tid`:

```
Name                | oldtid | newtid

KDDCup09-Upselling  | 360115 | 360975
MIP-2016-regression | 359947 | 360945
QSAR-TID-10980      |  14097 | 360933
QSAR-TID-11         |  13854 | 360932
```

### task_metadata_new_and_old.csv

This is a union of `task_metadata.csv` and `task_metadata_old.csv` on the `tid` column to contain 108 rows.
