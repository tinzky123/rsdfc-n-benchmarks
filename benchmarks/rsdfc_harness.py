rsdfc_harness.py -- correctness + performance harness for the rsdfc_rs binary.

    python3 rsdfc_harness.py [path/to/rsdfc_rs] [--quick] [--real] [--no-fuzz] [-v]

Needs only python3 + numpy (xz comes from the standard library; zstd is used if the `zstd`
command is installed). Exit code is 0 only if every check passes, so it can gate a release
build or a GitHub Actions job.

What it checks, per dataset
  * lossless: decompress(compress(x)) == x, bit for bit
  * deterministic: compressed bytes are identical with RSDFC_THREADS=1 and with all cores
  * bounded: output never exceeds input + 16 bytes (the raw fallback)
  * ratio vs xz -9 (and zstd -19 if available), so every claim is measured against what a
    customer already has
  * compress / decompress throughput on 1 thread vs all cores
and then, unless --no-fuzz:
  * the decoder survives corrupted and truncated packages (clean error, never a crash or hang)

Datasets are grouped: typical single-column numeric data (the product's target), data shaped
like the bugs we have actually hit (-0.0, large files, short files), and data RSD is NOT meant
to win on (reported, but never counted as a failure).
"""
import argparse, csv, lzma, os, random, re, shutil, subprocess, sys, tempfile, time
import numpy as np

# --------------------------------------------------------------------------- datasets
def synthetic(quick):
    rng = np.random.default_rng(42)
    n = 60000
    i = np.arange(n)
    D = []  # (group, name, bytes, is_target)

    def add(group, name, arr, target=True):
        data = arr if isinstance(arr, (bytes, bytearray)) else np.ascontiguousarray(arr).tobytes()
        D.append((group, name, bytes(data), target))

    # typical single-column numeric data
    add("typical", "u32 counter", np.arange(n, dtype="<u4"))
    add("typical", "u32 timestamps (+jitter)", np.cumsum(1000 + rng.integers(-3, 4, n)).astype("<u4"))
    add("typical", "i16 sensor (sine+noise)", (20000 * np.sin(2 * np.pi * i / 500) + rng.integers(-8, 9, n)).astype("<i2"))
    add("typical", "i32 random walk", np.cumsum(rng.integers(-5, 6, n)).astype("<i4"))
    add("typical", "i64 slow ramp", (10**12 + np.cumsum(rng.integers(0, 40, n))).astype("<i8"))
    add("typical", "f32 smooth sine", (np.sin(np.arange(30000) * 0.002) * 50).astype("<f4"))
    add("typical", "f32 sensor 2 decimals", np.round(20 + np.cumsum(rng.normal(0, 0.05, 30000)), 2).astype("<f4"))
    add("typical", "f64 price walk (cents)", np.round(100 + np.cumsum(rng.normal(0, 0.05, 15000)), 2).astype("<f8"))

    # shaped like bugs we have hit
    p = np.round(np.cumsum(rng.normal(0, 1, 200000)), 2).astype("<f8")
    p[[10666, 90000]] = -0.0
    add("regression", "f64 cents with two -0.0", p)
    q = np.round(rng.normal(0, 2, 40000), 2).astype("<f4")
    q[rng.integers(0, q.size, 25)] = -0.0
    add("regression", "f32 decimals with many -0.0", q)
    big = 400_000 if quick else 2_000_000
    t = np.arange(big)
    add("regression", f"i16 sensor {big*2//1000} KB (block-split)",
        (8000 * np.sin(t / 37.0) + 3000 * np.sin(t / 510.0) + rng.normal(0, 25, big)).astype("<i2"))
    add("regression", f"i32 sensor {big*4//1000} KB (block-split)",
        (1e6 * np.sin(t / 900.0) + rng.normal(0, 300, big)).astype("<i4"))

    # edge cases: must round-trip and stay bounded; ratio is irrelevant
    add("edge", "empty file", b"", False)
    add("edge", "1 byte", b"\x07", False)
    add("edge", "63 bytes (below search size)", bytes(range(63)), False)
    add("edge", "i32 + 3 trailing bytes", np.arange(5000, dtype="<i4").tobytes() + b"\x01\x02\x03", False)
    add("edge", "constant i32", np.full(20000, 7, "<i4"), False)
    f = np.round(np.cumsum(rng.normal(0, 1, 5000)), 2).astype("<f8")
    f[[3, 4000]] = [np.nan, np.inf]
    add("edge", "f64 decimals with NaN/inf", f, False)
    add("edge", "i32 negatives (sine)", (-50000 + 40000 * np.sin(np.arange(20000) / 50.0)).astype("<i4"), False)

    # NOT the target: reported for honesty, never a failure
    add("not-target", "random bytes", rng.integers(0, 256, 100000, dtype=np.uint8), False)
    add("not-target", "ASCII text", (b"the quick brown fox jumps over the lazy dog. " * 2000), False)
    return D


def real_datasets():
    """Single columns from the CSVs real_world_bench.py caches (if you have run it)."""
    cache = os.path.expanduser("~/rsdfc_realdata")
    out = []
    def col(fn, c, dt, scale=None):
        path = os.path.join(cache, fn)
        if not os.path.exists(path):
            return
        with open(path, newline="") as fh:
            vals = [r[c] for r in csv.DictReader(fh) if r.get(c, "") != ""]
        out.append(("real", f"{fn.split('.')[0]}:{c} ({dt[1:]})", np.array([float(v) for v in vals]).astype(dt).tobytes(), True))
    col("diamonds.csv", "price", "<i4")
    col("diamonds.csv", "carat", "<f8")
    col("seattle-weather.csv", "temp_max", "<f8")
    col("sp500.csv", "SP500", "<f8")
    col("taxis.csv", "fare", "<f8")
    return out

# --------------------------------------------------------------------------- tools
def run(cmd, threads=None, timeout=300):
    env = dict(os.environ)
    if threads is not None:
        env["RSDFC_THREADS"] = str(threads)
    else:
        env.pop("RSDFC_THREADS", None)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    return r, time.perf_counter() - t0

def zstd_size(path):
    if not shutil.which("zstd"):
        return None
    r = subprocess.run(["zstd", "-19", "-q", "-c", path], capture_output=True)
    return len(r.stdout) if r.returncode == 0 else None

def mbps(nbytes, secs):
    return nbytes / 1e6 / secs if secs > 0 else float("inf")

# --------------------------------------------------------------------------- main checks
def check_dataset(binp, name, data, tmp, verbose):
    """Returns (ok, row dict, packed bytes)."""
    inp = os.path.join(tmp, "in.bin")
    open(inp, "wb").write(data)
    problems = []
    packs, times = {}, {}
    for label, th in (("1t", 1), ("all", None)):
        out = os.path.join(tmp, f"c_{label}.rsd")
        r, dt = run([binp, "compress", inp, out], th)
        if r.returncode != 0:
            return False, {"err": "compress failed: " + r.stderr.strip()[:60]}, b""
        packs[label] = open(out, "rb").read()
        times[label] = dt
        if label == "1t":
            m = re.search(r"\[(.*)\]\s+[\d.]+s\s*$", r.stdout.strip())
            desc = m.group(1) if m else "raw"
    if packs["1t"] != packs["all"]:
        problems.append("output depends on thread count")
    pk = packs["1t"]
    if len(pk) > len(data) + 16:
        problems.append(f"expanded by {len(pk) - len(data)} bytes")

    dtimes = {}
    for label, th in (("1t", 1), ("all", None)):
        rec = os.path.join(tmp, f"d_{label}.bin")
        r, dt = run([binp, "decompress", os.path.join(tmp, f"c_{label}.rsd"), rec], th)
        dtimes[label] = dt
        if r.returncode != 0:
            problems.append("decompress failed: " + r.stderr.strip()[:50])
        elif open(rec, "rb").read() != data:
            problems.append(f"ROUND-TRIP MISMATCH ({label})")

    xz = len(lzma.compress(data, preset=9)) if data else 0
    zs = zstd_size(inp) if data else None
    row = {
        "orig": len(data), "comp": len(pk),
        "ratio": len(data) / len(pk) if pk else 0,
        "xz": len(data) / xz if xz else 0,
        "zstd": len(data) / zs if zs else None,
        "fmt": f"v{pk[4]}" if len(pk) > 4 else "?",
        "c1": mbps(len(data), times["1t"]), "ca": mbps(len(data), times["all"]),
        "d1": mbps(len(data), dtimes["1t"]), "da": mbps(len(data), dtimes["all"]),
        "desc": desc if verbose else "",
        "problems": problems,
    }
    return not problems, row, pk


def fuzz_decoder(binp, packs, tmp, rounds):
    rnd = random.Random(1234)
    bad = total = 0
    path = os.path.join(tmp, "fz.rsd")
    for pk in packs:
        if len(pk) < 12:
            continue
        for _ in range(rounds):
            b = bytearray(pk)
            mode = rnd.choice(["flip", "flipmany", "trunc"])
            if mode == "trunc":
                b = b[:rnd.randrange(1, len(b))]
            else:
                for _ in range(1 if mode == "flip" else 8):
                    i = rnd.randrange(len(b)) if rnd.random() < 0.5 else rnd.randrange(min(len(b), 64))
                    b[i] ^= 1 << rnd.randrange(8)
            open(path, "wb").write(b)
            total += 1
            try:
                r, _ = run([binp, "decompress", path, os.path.join(tmp, "fz.out")], 2, timeout=30)
                if r.returncode not in (0, 1):
                    bad += 1
                    print(f"  decoder CRASH rc={r.returncode}: {r.stderr.strip()[-90:]}")
            except subprocess.TimeoutExpired:
                bad += 1
                print("  decoder HANG on corrupted package")
    return total, bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("binary", nargs="?", default="./target/release/rsdfc_rs")
    ap.add_argument("--quick", action="store_true", help="smaller large-file cases")
    ap.add_argument("--real", action="store_true", help="add single columns from ~/rsdfc_realdata")
    ap.add_argument("--no-fuzz", action="store_true", help="skip corrupted-package tests")
    ap.add_argument("-v", "--verbose", action="store_true", help="show chosen strategy per file")
    a = ap.parse_args()
    if not os.path.exists(a.binary):
        sys.exit(f"binary not found: {a.binary}\nbuild it first:  cargo build --release")

    sets = synthetic(a.quick) + (real_datasets() if a.real else [])
    print(f"binary: {a.binary}   cores: {os.cpu_count()}   zstd: {'yes' if shutil.which('zstd') else 'not installed'}")
    hdr = f"{'dataset':38}{'size':>10}{'ratio':>8}{'xz-9':>7}" + (f"{'zstd19':>8}" if shutil.which('zstd') else "") \
          + f"{'fmt':>4}{'comp MB/s 1t':>14}{'all':>7}{'dec MB/s 1t':>13}{'all':>7}  status"
    fails, last_group, packs = [], None, []
    losses_vs_xz = []
    with tempfile.TemporaryDirectory() as tmp:
        for group, name, data, target in sets:
            if group != last_group:
                print(f"\n== {group}\n{hdr}\n" + "-" * len(hdr))
                last_group = group
            ok, row, pk = check_dataset(a.binary, name, data, tmp, a.verbose)
            if "err" in row:
                print(f"{name:38}  {row['err']}")
                fails.append(name)
                continue
            if pk:
                packs.append(pk)
            if target and row["ratio"] < row["xz"]:
                losses_vs_xz.append(name)
            z = f"{row['zstd']:7.1f}x" if row["zstd"] else (f"{'-':>8}" if shutil.which('zstd') else "")
            status = "ok" if ok else "FAIL: " + "; ".join(row["problems"])
            print(f"{name:38}{row['orig']:10,}{row['ratio']:7.1f}x{row['xz']:6.1f}x{z}{row['fmt']:>4}"
                  f"{row['c1']:14.1f}{row['ca']:7.1f}{row['d1']:13.1f}{row['da']:7.1f}  {status}"
                  + (f"  {row['desc']}" if a.verbose else ""))
            if not ok:
                fails.append(name)

        if not a.no_fuzz:
            print("\n== corrupted / truncated packages (decoder must fail cleanly)")
            total, bad = fuzz_decoder(a.binary, packs[:20], tmp, 12)
            print(f"{total} corrupted packages, {bad} crashes/hangs")
            if bad:
                fails.append("decoder fuzz")

    print("\n" + "=" * 60)
    if losses_vs_xz:
        print("note: RSD ratio below xz-9 on target data: " + ", ".join(losses_vs_xz))
    print(f"{len(sets) - len([f for f in fails if f != 'decoder fuzz'])}/{len(sets)} datasets passed"
          + ("" if not fails else f"   FAILURES: {', '.join(fails)}"))
    print("ALL CHECKS PASSED" if not fails else "SOME CHECKS FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
