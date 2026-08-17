# Paper experiment wrapper

`run_paper_serial.sh` executes paper-profile experiments as independent seed
shards. Each runner checkpoints completed episode rows, so rerunning the same
command resumes incomplete shards instead of repeating successful cells.

Run commands from the repository root:

```bash
./paper_experiments_v3/scripts/run_paper_serial.sh p02
./paper_experiments_v3/scripts/run_paper_serial.sh p05
```

Inspect completion status without launching new work:

```bash
./paper_experiments_v3/scripts/run_paper_serial.sh core --status
```

Select an explicit seed subset when reproducing or repairing a shard:

```bash
./paper_experiments_v3/scripts/run_paper_serial.sh p04 --seeds 1,2,3
```

On a machine with sufficient memory, run seed shards concurrently:

```bash
./paper_experiments_v3/scripts/run_paper_serial.sh p02 p03 p06 --jobs 6
```

Parallel child-process logs are written under
`paper_experiments_v3/outputs/_runner_logs/`. P05 emits large diagnostic event
logs incrementally; run it with conservative parallelism. These experiments
primarily require RAM, CPU, disk, and stable API access. GPU memory matters only
when the backend is changed to local model inference.

Available aliases:

- `core`: P02, P03, P04, P05
- `appendix`: P06, P07
- `all`: P01--P07
