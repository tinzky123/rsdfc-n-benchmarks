# Running the RSDFC/N Benchmark Harness

## Requirements

- Linux x86-64 (the current product target)
- Python 3
- NumPy
- The compatible `rsdfc_rs` executable
- Optional: the `zstd` command for Zstandard comparisons

## Install the Python dependency

```bash
python3 -m pip install numpy
```

## Run the quick tests

From the repository root, with the compatible executable available at `./target/release/rsdfc_rs`:

```bash
python3 benchmarks/rsdfc_harness.py --quick
```

## Run the full tests

```bash
python3 benchmarks/rsdfc_harness.py
```

The full run includes larger synthetic datasets and corrupted/truncated package tests.

If the executable is located elsewhere, pass its path as the first argument:

```bash
python3 benchmarks/rsdfc_harness.py /path/to/rsdfc_rs --quick
```

## What is measured?

- Lossless compression and decompression
- Output consistency between single-threaded and default-thread runs
- Compression ratios compared with XZ-9 and, when installed, Zstandard-19
- Compression and decompression throughput
- Behavior on corrupted and truncated compressed packages

## Reporting results

Record the operating system, CPU, executable version, installed Zstandard version, exact command, and complete output when sharing results.

Do not treat results from one dataset as representative of all files. The harness generates synthetic data; results should be independently reproduced before being presented as verified claims.
