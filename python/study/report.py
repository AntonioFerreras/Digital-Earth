"""The encoding study's results table and Pareto plots from cache/study/runs/*/result.json,
joined with the inference benchmark's timings."""

import json
import math
import re
from pathlib import Path


# Reference palette slots 1-6 (dataviz skill); markers repeat the identity for colour-blind readers.
FAMILY_STYLE = {
    "A": ("#2a78d6", "o", "A plain MLP"),
    "B": ("#eb6834", "s", "B Fourier + MLP"),
    "C": ("#1baf7a", "^", "C hash grid + head"),
    "D": ("#eda100", "D", "D K-Planes + head"),
    "E": ("#e87ba4", "v", "E fitted dense grid"),
    "F": ("#008300", "*", "F baked LUT"),
}
BUDGET_MS = 10.0


def load_results(runs_dir: Path) -> list[dict]:
    results = [json.loads(path.read_text()) for path in sorted(runs_dir.glob("*/result.json"))]
    assert results, f"no results in {runs_dir}"
    return results


def load_timings(bench_dir: Path) -> list[dict]:
    """Every benchmark row of the bench's results*.json files, newest file last."""
    rows = []
    for path in sorted(bench_dir.glob("results*.json"), key=lambda p: p.stat().st_mtime):
        rows += json.loads(path.read_text())["results"]
    return rows


def timing_for(result: dict, timings: list[dict]) -> tuple[dict | None, bool]:
    """The fastest benchmarked variant of a run's configuration, and whether it is a stand-in.

    Bench names are the config's tokens in order with variant tokens (precision,
    weight source, storage, plain-MLP activation) between them. Tables time the
    same at any size or cell schedule, so a coarse hash grid takes its bench
    sibling's row and a dense table the LUT row closest in size."""
    spec = result["spec"]
    kind = spec["encoding"]["kind"]
    if kind in ("dense", "bruneton", "baked"):
        rows = [row for row in timings if row["name"].startswith("lut_") and row["name"].endswith("_half_filtered")]
        if not rows:
            return None, False
        row = min(rows, key=lambda row: abs(math.log(row["parameters"] / result["parameters"])))
        return row, row["parameters"] != result["parameters"]
    tokens = spec["config"].split("_")
    stand_in = "coarse" in tokens
    tokens = [token for token in tokens if token != "coarse"]
    if kind == "raw":
        tokens.insert(2, spec["head"]["activation"])
    pattern = re.compile("^" + "".join(("" if i == 0 else "_") + re.escape(t) + r"(_[^_]+)*?"
                                       for i, t in enumerate(tokens)) + "$")
    matches = [row for row in timings if pattern.match(row["name"])]
    if not matches:
        return None, False
    return min(matches, key=lambda row: max(row["space"]["ms"], row["limb"]["ms"])), stand_in


def timing_ms(row: dict) -> float:
    return max(row["space"]["ms"], row["limb"]["ms"])


def is_sweep_row(result: dict) -> bool:
    """A configuration at the study's default settings: 16k data, relative loss, identity
    output, seed 0, default budget, u_mu rotation for every encoding but the split-mu tables."""
    spec = result["spec"]
    if spec["candidate"] == "F":
        return True
    suffix = "_norot" if spec["encoding"]["kind"] == "bruneton" else ""
    return spec["name"] == spec["config"] + suffix


def bytes_text(parameters: int) -> str:
    return f"{parameters * 2 / 2**20:.2f} / {parameters * 4 / 2**20:.2f}"


def table(results: list[dict], timings: list[dict], robustness: dict[str, float]) -> str:
    """Markdown rows sorted by family then error; `robustness` maps a config to its 1k/16k error ratio."""
    lines = [
        "| cand. | run | params | MB fp16 / fp32 | samples seen | RMSE | mean rel. (lit) R / G / B | "
        "p99 rel. | split | horizon | terminator | forward | sunset | 1k/16k | ms (bench variant) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in sorted(results, key=lambda r: (r["spec"]["candidate"], r["metrics"]["rel_mean"])):
        spec, m = result["spec"], result["metrics"]
        timing, stand_in = timing_for(result, timings)
        ms = "n/a" if timing is None else f"{'~' if stand_in else ''}{timing_ms(timing):.2f} (`{timing['name']}`)"
        ratio = robustness.get(spec["name"])
        lines.append(
            f"| {spec['candidate']} | `{spec['name']}` | {result['parameters']:,} | {bytes_text(result['parameters'])} | "
            f"{result['samples_seen'] / 1e9:.2f} G | {m['rmse']:.3g} | "
            f"{100 * m['rel_mean']:.2f}% ({100 * m['rel_mean_r']:.1f} / {100 * m['rel_mean_g']:.1f} / {100 * m['rel_mean_b']:.1f}) | "
            f"{100 * m['rel_p99']:.1f}% | {100 * m['rel_mean_split']:.1f}% | {100 * m['rel_mean_horizon']:.2f}% | {100 * m['rel_mean_terminator']:.2f}% | "
            f"{100 * m['rel_mean_forward']:.2f}% | {100 * m['rel_mean_sunset']:.2f}% | "
            f"{'' if ratio is None else f'{ratio:.2f}'} | {ms} |")
    return "\n".join(lines)


def pareto_plots(results: list[dict], timings: list[dict], out_dir: Path) -> list[Path]:
    """Mean relative error (lit) against parameters and against benchmark ms, one PNG each."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    paths = []
    for axis_name in ("parameters", "ms"):
        figure, axis = pyplot.subplots(figsize=(6.4, 4.0))
        for family, (color, marker, label) in FAMILY_STYLE.items():
            points = []
            for result in results:
                spec = result["spec"]
                if spec["candidate"] != family:
                    continue
                if axis_name == "parameters":
                    x = result["parameters"]
                else:
                    timing, _ = timing_for(result, timings)
                    if timing is None:
                        continue
                    x = timing_ms(timing)
                points.append((x, 100 * result["metrics"]["rel_mean"]))
            if points:
                xs, ys = zip(*points)
                axis.scatter(xs, ys, s=90 if marker == "*" else 36, color=color, marker=marker, label=label, edgecolors="white", linewidths=1.0)
        axis.set_xscale("log")
        axis.set_yscale("log")
        if axis_name == "ms":
            axis.axvline(BUDGET_MS, color="#52514e", linewidth=1, linestyle="--")
            axis.text(BUDGET_MS, axis.get_ylim()[1], " 10 ms budget", va="top", fontsize=8, color="#52514e")
        axis.set_xlabel("parameters" if axis_name == "parameters" else "ms per frame, 2560x1440, RTX 4090 (worse of space/limb view)")
        axis.set_ylabel("mean relative error, lit points (%)")
        axis.grid(True, which="major", color="#e4e3df", linewidth=0.6)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.legend(fontsize=7, frameon=False)
        figure.tight_layout()
        path = out_dir / f"20261019-encoding-pareto-{axis_name}.png"
        figure.savefig(path, dpi=110)
        pyplot.close(figure)
        paths.append(path)
    return paths
