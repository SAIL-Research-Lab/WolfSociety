# Paper shard runners

Use these wrappers for paper-profile runs after the memory crash. Each command
runs one seed shard at a time. The underlying runner checkpoints each completed
episode to `rows.jsonl`, so interrupted shards resume at the missing/error
cells instead of repeating every episode in the seed.

Examples:

```bash
cd /Users/zhangyuejun/Documents/aaai/WolfBench-main

./paper_experiments_v3/scripts/run_paper_serial.sh p02
./paper_experiments_v3/scripts/run_paper_serial.sh p05
```

Running two terminals is reasonable. Avoid running more than one P05 terminal
because P05 writes large decision/message/exposure diagnostics.

## Local-safe plan

Use this plan on a laptop or desktop after a memory crash. It prioritizes not
crashing over total wall-clock time.

First, check progress:

```bash
cd /Users/zhangyuejun/Documents/aaai/WolfBench-main
./paper_experiments_v3/scripts/run_paper_serial.sh remaining --status
```

Then run in this order:

```bash
# Terminal 1 only. P03 is the shortest missing paper experiment.
./paper_experiments_v3/scripts/run_paper_serial.sh p03

# After P03 completes, run P02.
./paper_experiments_v3/scripts/run_paper_serial.sh p02

# After P02 completes, run appendix role robustness.
./paper_experiments_v3/scripts/run_paper_serial.sh p06

# Run P05 alone. Do not run any other experiment while this is active.
./paper_experiments_v3/scripts/run_paper_serial.sh p05

# Last, repair the P04 network/API-error shards.
./paper_experiments_v3/scripts/run_paper_serial.sh p04 --seeds 5,7,8,9,10,11,12
```

If you are comfortable using two terminals locally, only pair P03 with P02 or
P06. Do not pair anything with P05. Do not use `--jobs` locally unless you have
verified memory headroom.

On a server, use `--jobs` to run multiple seed shards concurrently:

```bash
# Moderate server, e.g. 64 GB RAM.
./paper_experiments_v3/scripts/run_paper_serial.sh p02 p03 p06 --jobs 6

# Larger server, e.g. 128 GB RAM.
./paper_experiments_v3/scripts/run_paper_serial.sh p05 --jobs 4
```

Parallel child-process output is written to:

```bash
paper_experiments_v3/outputs/_runner_logs/
```

P05 writes large diagnostic event logs incrementally to JSONL and converts them
to CSV at the end of the shard. This lowers peak memory and preserves completed
episodes if the process is interrupted.

These experiments mainly need RAM, CPU, disk, and stable network/API access.
GPU VRAM is not the limiting resource unless the backend is changed to local
model inference.

Useful commands:

```bash
# Show what remains for the core experiments.
./paper_experiments_v3/scripts/run_paper_serial.sh core --status

# Repair only the P04 seeds that previously had network/API errors.
./paper_experiments_v3/scripts/run_paper_serial.sh p04 --seeds 5,7,8,9,10,11,12

# Repair those P04 shards in parallel on a server.
./paper_experiments_v3/scripts/run_paper_serial.sh p04 --seeds 5,7,8,9,10,11,12 --jobs 4

# Run appendix role robustness after the core experiments.
./paper_experiments_v3/scripts/run_paper_serial.sh p06
```

Aliases:

- `core`: P02, P03, P04, P05
- `remaining`: P02, P03, P04, P05, P06
- `appendix`: P06, P07
- `all`: P01--P07
