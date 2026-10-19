#!/usr/bin/env python3
"""Encoding study for the MLP atmosphere mode: train the configurations of
study/configs.py under one wall-clock budget, and evaluate the baked LUT.

Results land in cache/study/runs/<run>/result.json; see
docs/research/20261019-neural-encoding-results.md. python/export_weights.py --run
exports a run for the renderer's neural mode.

Usage:
    python python/encoding_study.py train hash12_w16x1_t18 [--data 1k] [--loss mse] [--output softplus]
    python python/encoding_study.py baked-lut
    python python/encoding_study.py report
    python python/encoding_study.py render kplanes16_w16x1_s16 [...]
"""

import argparse
import json
from pathlib import Path

import numpy as np

DATA_DIR = Path("cache/training")
RUNS_DIR = Path("cache/study/runs")
DEFAULT_BUDGET_S = 300.0
FIGURES_DIR = Path("docs/research/figures")
# The inference benchmark's worktree, a sibling of this one.
BENCH_DIR = Path("../zs-neural-bench")


def run_name(config: str, args: argparse.Namespace) -> str:
    """The config name, plus every setting that differs from the study's default."""
    parts = [config]
    if args.data != "16k":
        parts.append(f"data{args.data}")
    if args.loss != "rel":
        parts.append(args.loss)
    if args.output != "identity":
        parts.append(args.output)
    if not args.mu_rotation:
        parts.append("norot")
    if args.tv > 0:
        parts.append(f"tv{args.tv:g}")
    if args.budget_s != DEFAULT_BUDGET_S:
        parts.append(f"b{args.budget_s:g}")
    if args.seed != 0:
        parts.append(f"seed{args.seed}")
    if args.lr_scale != 1.0:
        parts.append(f"lr{args.lr_scale:g}")
    return "_".join(parts)


def train(args: argparse.Namespace) -> None:
    from study import configs, training

    spec = dict(configs.specs()[args.config])
    spec.update(name=run_name(args.config, args), config=args.config, data=args.data, loss=args.loss,
                output=args.output, tv=args.tv,
                mu_rotation=args.mu_rotation)
    for key in ("lr_grid", "lr_head"):
        if key in spec:
            spec[key] *= args.lr_scale
    training.run(spec, DATA_DIR, RUNS_DIR / spec["name"], args.budget_s, args.batch_size, args.seed)


def baked_lut(args: argparse.Namespace) -> None:
    from study import baked

    result, prediction = baked.evaluate(DATA_DIR)
    out_dir = RUNS_DIR / result["spec"]["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "ref_prediction.npy", prediction)
    (out_dir / "result.json").write_text(json.dumps(result, indent=1))
    for key in ("metrics", "metrics_shipped_texture"):
        print(key, json.dumps(result[key], indent=1))


def report(args: argparse.Namespace) -> None:
    """Rescores every run from its saved reference prediction, then prints the table and writes the plots."""
    import datagen_format
    from study import metrics, report as study_report

    reference = datagen_format.read_records(DATA_DIR / "ref_clean.edgn")
    results = study_report.load_results(RUNS_DIR)
    for result in results:
        prediction = np.load(RUNS_DIR / result["spec"]["name"] / "ref_prediction.npy")
        result["metrics"] = metrics.evaluate(prediction, reference)
    by_name = {result["spec"]["name"]: result for result in results}
    robustness = {}
    for name, result in by_name.items():
        if result["spec"].get("data") == "1k":
            clean = by_name[name.replace("_data1k", "")]
            robustness[clean["spec"]["name"]] = result["metrics"]["rel_mean"] / clean["metrics"]["rel_mean"]
    timings = study_report.load_timings(args.bench / "bench-neural")
    print(study_report.table(results, timings, robustness))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for path in study_report.pareto_plots([r for r in results if study_report.is_sweep_row(r)], timings, FIGURES_DIR):
        print(f"wrote {path}")


def shipped(args: argparse.Namespace) -> None:
    """Scores runs at shipped precision; writes shipped.json next to each result."""
    import datagen_format
    from study import metrics, render, shipped as study_shipped

    reference = datagen_format.read_records(DATA_DIR / "ref_clean.edgn")
    for name in args.runs:
        run_dir = RUNS_DIR / name
        spec = json.loads((run_dir / "result.json").read_text())["spec"]
        model = render.load_model(run_dir)
        scores = {storage: metrics.evaluate(study_shipped.predict(model, spec, reference[:, :4], storage), reference)
                  for storage in ("texture", "buffer")}
        (run_dir / "shipped.json").write_text(json.dumps(scores, indent=1))
        print(f"{name}: rel_mean texture {scores['texture']['rel_mean']:.4f}, buffer {scores['buffer']['rel_mean']:.4f}")


def render_runs(args: argparse.Namespace) -> None:
    from study import render

    errors = render.compare([RUNS_DIR / name for name in args.runs], args.views, Path("cache/study/renders"), DATA_DIR,
                            FIGURES_DIR / args.out)
    print(json.dumps(errors, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="train one configuration")
    train_parser.add_argument("config")
    train_parser.add_argument("--data", choices=("16k", "1k", "lut"), default="16k")
    train_parser.add_argument("--loss", choices=("rel", "mse"), default="rel")
    train_parser.add_argument("--output", choices=("identity", "softplus"), default="identity")
    train_parser.add_argument("--no-mu-rotation", dest="mu_rotation", action="store_false",
                              help="feed u_mu as is instead of frac(u_mu + 0.5)")
    train_parser.add_argument("--tv", type=float, default=0.0, help="total-variation weight (dense grids)")
    train_parser.add_argument("--lr-scale", type=float, default=1.0)
    train_parser.add_argument("--budget-s", type=float, default=DEFAULT_BUDGET_S)
    train_parser.add_argument("--batch-size", type=int, default=2**17)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.set_defaults(handler=train)
    commands.add_parser("baked-lut", help="evaluate the baked LUT on the reference points").set_defaults(handler=baked_lut)
    report_parser = commands.add_parser("report", help="results table and Pareto plots")
    report_parser.add_argument("--bench", type=Path, default=BENCH_DIR, help="the inference benchmark's worktree")
    report_parser.set_defaults(handler=report)
    shipped_parser = commands.add_parser("shipped", help="score runs at shipped precision")
    shipped_parser.add_argument("runs", nargs="+")
    shipped_parser.set_defaults(handler=shipped)
    render_parser = commands.add_parser("render", help="render runs next to the path tracer")
    render_parser.add_argument("runs", nargs="+")
    render_parser.add_argument("--views", nargs="+", default=["default", "sunset_limb"])
    render_parser.add_argument("--out", default="20261019-encoding-renders.png")
    render_parser.set_defaults(handler=render_runs)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
