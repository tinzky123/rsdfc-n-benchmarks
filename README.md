# RSDFC/N Harness Edition — Public Benchmark

RSDFC/N is a numerical-data compression project written in Rust. This repository contains benchmark documentation, test instructions, and reproducibility materials for evaluating its performance on numerical datasets.

## Benchmark highlight

In one reported test using an unsigned 32-bit counter dataset, the Harness Edition achieved approximately **2,087× compression**.

The reported comparison for that dataset was:

| Compressor | Reported compression ratio |
|---|---:|
| RSDFC/N Harness Edition | ~2,087× |
| XZ-9 | ~14.4× |
| Zstandard-19 | ~1.6× |

**Important:** These are dataset-specific results, not a claim that RSDFC/N achieves these ratios on arbitrary files. Independent reproduction is encouraged.

## What this repository should contain

- Benchmark scripts and exact commands
- Dataset descriptions or synthetic test-data generators
- Input and output byte counts
- Compressor versions and settings
- Hardware and operating-system details
- Test results and known limitations

## Reproducing the results

Reproduction instructions and the required benchmark files will be added here after they have been checked against the published release.

## Scope

This repository is for benchmark transparency and evaluation. It does not imply that the complete RSDFC/N implementation is open source.

## Commercial release

RSDFC/N Harness Edition is available here:

https://mcewan0.gumroad.com/l/vawbsw
