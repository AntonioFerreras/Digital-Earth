"""The cloud bench's validation suite (TESTING.md).

`gt` renders the ground truth into validation/gt: per view the path tracer's image, the
converged probe field, and per view that field rendered with
the default config, converged and at the test's samples. It refuses to replace an
existing ground truth without --replace; replacing it needs the owner's approval.

`run` bakes a test field with a config's cloud bench settings, scores it against the
ground truth's field, renders it with the config's settings and with the default ones,
and scores the images against the ground truth's renders of the converged field.
Results land in validation/runs/<time>-<label>/results.json and validation/history.jsonl.

Run from the repository root, after `pixi run zig build -Doptimize=ReleaseFast`:
  pixi run -e py python python/validation/validate.py gt
  pixi run -e py python python/validation/validate.py run --config my_settings.zon --label photons
"""

import argparse
import datetime
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import metrics

EARTH = Path("zig-out/bin/earth")
SUITE = Path("tests/validation/suite.json")
DATA = Path("validation")

# Cloud bench settings that decide the field: a test bakes with its own, and its
# images with the default config keep them, as the field holds them.
BAKE_FIELDS = ("probe_layout", "probe_budget", "probe_depth_ratio", "probe_bake", "probe_sun_estimate", "probe_rounds_per_frame")
# Set by the suite: the scene stays the ground truth's, as only that compares.
SUITE_FIELDS = ("view", "probe_walks", "extinction", "bounce_limit", "sky")


def earth(*args: str) -> str:
    result = subprocess.run([str(EARTH), *args], capture_output=True, text=True)
    output = result.stdout + result.stderr
    if result.returncode != 0:
        raise SystemExit(f"earth {' '.join(args)} failed:\n{output}")
    return output


def bench_fields(config: Path) -> dict[str, str]:
    """The `.cloud_bench` block of a viewer config, values as written."""
    text = config.read_text()
    block = re.search(r"\.cloud_bench = \.\{(.*?)\n    \}", text, re.S)
    return dict(re.findall(r"\.(\w+) = ([^,\n]+),", block.group(1))) if block else {}


def write_config(view: Path, bench: dict[str, str], path: Path) -> Path:
    """`view` with the cloud bench settings `bench`, the rest at their defaults."""
    lines = "".join(f"        .{name} = {value},\n" for name, value in bench.items())
    text = view.read_text().rstrip()
    assert text.endswith("}")
    path.write_text(text[:-1] + f"    .cloud_bench = .{{\n{lines}    }},\n}}\n")
    return path


def render(jobs: list[tuple[Path, Path, str, int]], size: int, probes: Path | None, jobs_path: Path,
           log: bool = False) -> dict[str, float] | str:
    """Renders each job's (config, out, bench view, spp) as `out`.png and .pfm, on one
    renderer with the field `probes`, and returns each out's milliseconds, or with `log`
    the command's output."""
    entries = "".join(f'    .{{ .config = "{config}", .out = "{out.with_suffix(".png")}", .out_linear = "{out.with_suffix(".pfm")}", '
                      f".bench_view = .{view}, .spp = {spp} }},\n" for config, out, view, spp in jobs)
    jobs_path.write_text(f".{{\n{entries}}}\n")
    field = ["--probes", str(probes)] if probes is not None else []
    output = earth("render", "--config", str(jobs[0][0]), "--jobs", str(jobs_path), *field, "--width", str(size), "--height", str(size))
    if probes is not None and not re.search(r"probes: [1-9]\d* walks from", output):
        raise SystemExit(f"{probes} did not load:\n{output}")
    times = {Path(out).stem: float(ms) for ms, out in re.findall(r"spp: ([\d.]+) ms \(.*?\) -> (\S+)", output)}
    missing = [out.stem for _, out, _, _ in jobs if out.stem not in times]
    if missing:
        raise SystemExit(f"not rendered: {missing}\n{output}")
    return output if log else times


def bake(config: Path, out: Path, half: Path | None = None) -> dict:
    args = ["bake-probes", "--config", str(config), "--out", str(out)]
    if half is not None:
        args += ["--half", str(half)]
    output = earth(*args)
    found = re.search(r"bake-probes: (\d+) walks in ([\d.]+) ms", output)
    return {"walks": int(found.group(1)), "bake_ms": float(found.group(2))}


def views(suite: dict) -> list[Path]:
    return sorted(Path(suite["views"]).glob("view_*.zon"))


def git_commit() -> str:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout
    return head + ("+dirty" if dirty else "")


def ground_truth(suite: dict, data: Path, replace: bool) -> None:
    gt_dir = data / "gt"
    if gt_dir.exists():
        if not replace:
            raise SystemExit(f"{gt_dir} exists; --replace it only with the owner's approval (TESTING.md)")
        shutil.rmtree(gt_dir)
    gt_dir.mkdir(parents=True)
    gt, size = suite["gt"], suite["size"]
    manifest: dict = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "commit": git_commit(), "suite": suite, "views": {}}
    field = gt_dir / "field.eprb"
    field_bench = {"probe_walks": str(gt["field_walks"]), "probe_rounds_per_frame": "64"}
    first = views(suite)[0]
    manifest["field"] = bake(write_config(first, field_bench, gt_dir / "field_bake.zon"), field)
    print(f"field: {manifest['field']}", flush=True)
    jobs = []
    for view in views(suite):
        config = write_config(view, field_bench, gt_dir / f"{view.stem}.zon")
        jobs += [(config, gt_dir / f"pt_{view.stem}", "reference", gt["pt_spp"]),
                 (config, gt_dir / f"field_{view.stem}", "approximation", gt["field_spp"]),
                 (config, gt_dir / f"floor_{view.stem}", "approximation", suite["test"]["spp"])]
    times = render(jobs, size, field, gt_dir / "jobs.zon")
    for view in views(suite):
        name = view.stem
        entry = {f"{kind}_ms": times[f"{kind}_{name}"] for kind in ("pt", "field", "floor")}
        entry["pt_vs_field"] = metrics.image_metrics(gt_dir / f"field_{name}", gt_dir / f"pt_{name}")
        entry["render_noise_floor"] = metrics.image_metrics(gt_dir / f"floor_{name}", gt_dir / f"field_{name}")
        manifest["views"][name] = entry
        print(f"{name}: {json.dumps(entry)}", flush=True)
    (gt_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def test_run(suite: dict, data: Path, config_path: Path, label: str, walks: int | None) -> None:
    gt_dir = data / "gt"
    manifest = json.loads((gt_dir / "manifest.json").read_text())
    test_bench = bench_fields(config_path)
    walks = walks or int(test_bench.get("probe_walks", manifest["suite"]["test"]["walks"]))
    run = data / "runs" / f"{datetime.datetime.now():%Y%m%d_%H%M%S}-{label}"
    run.mkdir(parents=True)
    size, spp = suite["size"], suite["test"]["spp"]
    rendered = {k: v for k, v in test_bench.items() if k not in SUITE_FIELDS} | {"probe_walks": str(walks)}
    defaults = {k: v for k, v in test_bench.items() if k in BAKE_FIELDS} | {"probe_walks": str(walks)}
    results: dict = {"label": label, "config": str(config_path), "commit": git_commit(), "walks": walks,
                     "test_bench": test_bench, "gt_created": manifest["created"], "views": {}}

    field, half = run / "field.eprb", run / "half.eprb"
    started = time.monotonic()
    results["bake"] = bake(write_config(views(suite)[0], rendered, run / "bake.zon"), field, half)
    truth, test, test_half = metrics.Field(gt_dir / "field.eprb"), metrics.Field(field), metrics.Field(half)
    results["convergence"] = metrics.convergence(test_half, test)
    if test.layout == truth.layout:
        results["field"] = metrics.field_metrics(test, truth)
    else:
        results["field"] = None
        print("field metrics skipped: the test's layout or medium differs from the ground truth's")
    print(f"bake: {results['bake']}\nconvergence: {results['convergence']}\nfield: {results['field']}", flush=True)

    jobs = []
    for view in views(suite):
        jobs += [(write_config(view, rendered, run / f"{view.stem}_test.zon"), run / f"test_{view.stem}", "approximation", spp),
                 (write_config(view, defaults, run / f"{view.stem}_default.zon"), run / f"default_{view.stem}", "approximation", spp)]
    times = render(jobs, size, field, run / "jobs.zon")
    for view in views(suite):
        name = view.stem
        truth_image = gt_dir / f"field_{name}"
        entry = {"test_ms": times[f"test_{name}"], "default_ms": times[f"default_{name}"]}
        entry["test_vs_gt"] = metrics.image_metrics(run / f"test_{name}", truth_image)
        entry["default_vs_gt"] = metrics.image_metrics(run / f"default_{name}", truth_image)
        entry["pt_vs_gt"] = manifest["views"][name]["pt_vs_field"]
        entry["render_noise_floor"] = manifest["views"][name]["render_noise_floor"]
        results["views"][name] = entry
        print(f"{name}: test {summary(entry['test_vs_gt'])} | default {summary(entry['default_vs_gt'])}", flush=True)
    results["wall_s"] = time.monotonic() - started
    (run / "results.json").write_text(json.dumps(results, indent=2))
    with open(data / "history.jsonl", "a") as history:
        history.write(json.dumps({k: results[k] for k in ("label", "commit", "walks", "bake", "convergence", "field")}
                                 | {"means": means(results)}) + "\n")
    print_report(results)


def summary(m: dict) -> str:
    return f"PSNR {m['psnr_db']:.2f} dB, SSIM {m['ssim']:.4f}"


def means(results: dict) -> dict:
    keys = ("test_vs_gt", "default_vs_gt", "pt_vs_gt", "render_noise_floor")
    out = {}
    for key in keys:
        rows = [v[key] for v in results["views"].values()]
        out[key] = {m: sum(r[m] for r in rows) / len(rows) for m in ("psnr_db", "ssim")}
    return out


def print_report(results: dict) -> None:
    print(f"\n{results['label']}: {results['walks']} walks, bake {results['bake']['bake_ms'] / 1000:.2f} s")
    print(f"{'mean over views':22s} {'PSNR dB':>8} {'SSIM':>7}")
    for key, m in means(results).items():
        print(f"{key:22s} {m['psnr_db']:8.2f} {m['ssim']:7.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    gt = commands.add_parser("gt", help="render the ground truth")
    gt.add_argument("--replace", action="store_true", help="replace an existing ground truth (owner's approval only)")
    run = commands.add_parser("run", help="score a config's cloud bench settings")
    run.add_argument("--config", type=Path, required=True, help="viewer config whose .cloud_bench is tested")
    run.add_argument("--label", default="run")
    run.add_argument("--walks", type=int, help="walks or photon rounds to bake (default: the config's, else the suite's)")
    parser.add_argument("--suite", type=Path, default=SUITE)
    parser.add_argument("--data", type=Path, default=DATA, help="where the ground truth and the runs live")
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    if not EARTH.exists():
        raise SystemExit(f"{EARTH} missing: pixi run zig build -Doptimize=ReleaseFast")
    if args.command == "gt":
        ground_truth(suite, args.data, args.replace)
    else:
        test_run(suite, args.data, args.config, args.label, args.walks)


if __name__ == "__main__":
    sys.exit(main())
