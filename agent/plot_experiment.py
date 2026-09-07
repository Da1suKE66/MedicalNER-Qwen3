"""Static scientific plots from recorded results; no inferred training points."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--new", required=True)
    p.add_argument("--oracle", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    history = json.loads(Path(args.history).read_text())
    baseline = json.loads(Path(args.baseline).read_text())
    model = json.loads(Path(args.new).read_text())
    oracle = json.loads(Path(args.oracle).read_text())
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 13,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.7), layout="constrained")
    colors = ["#196c98", "#29936a"]
    epochs = sorted({r["epoch"] for r in history if r["event"] == "train"})
    for color, epoch in zip(colors, epochs):
        rows = [r for r in history if r["event"] == "train" and r["epoch"] == epoch]
        axs[0].plot(
            [r["step"] for r in rows],
            [r["loss"] for r in rows],
            color=color,
            label=f"Epoch {epoch}",
            lw=2,
        )
    axs[0].set(
        title="Relation training BCE",
        xlabel="Optimizer step",
        ylabel="Epoch-to-date mean loss",
    )
    axs[0].legend(frameon=False)
    axs[0].grid(alpha=0.15)
    tune = [r for r in history if r["event"] == "tune"]
    x = np.arange(len(tune))
    width = 0.24
    for offset, key, color in [
        (-width, "precision", "#196c98"),
        (0, "recall", "#dc8f34"),
        (width, "f1", "#29936a"),
    ]:
        bars = axs[1].bar(
            x + offset,
            [r["selected"][key] for r in tune],
            width,
            label=key.capitalize(),
            color=color,
        )
        axs[1].bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axs[1].set(
        xticks=x,
        xticklabels=[
            f"Epoch {r['epoch']}\nt={r['selected']['threshold']:.2f}" for r in tune
        ],
        ylim=(0, 1.1),
        title="Held-out training groups: tuning",
    )
    axs[1].legend(frameon=False, fontsize=9, loc="upper left")
    axs[1].grid(axis="y", alpha=0.15)
    reports = [baseline, model, oracle]
    x = np.arange(3)
    for offset, key, color in [
        (-width, "precision", "#196c98"),
        (0, "recall", "#dc8f34"),
        (width, "f1", "#29936a"),
    ]:
        bars = axs[2].bar(
            x + offset,
            [r["free_text"]["relation"][key] for r in reports],
            width,
            label=key.capitalize(),
            color=color,
        )
        axs[2].bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axs[2].set(
        xticks=x,
        xticklabels=[
            "v4 generator\n512-token cap",
            "New classifier\nsame v4 entities",
            "Classifier +\ngold entities",
        ],
        ylim=(0, 1.1),
        title="81 free-text records: regression",
    )
    axs[2].grid(axis="y", alpha=0.15)
    fig.suptitle(
        "Relation reliability experiment | Qwen3-8B LoRA | 2026-09-07", fontsize=16
    )
    fig.savefig(out / "training_and_evaluation.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 4.6), layout="constrained")
    for r, color in zip(tune, colors):
        sweep = r["sweep"]
        ax.plot(
            [s["recall"] for s in sweep],
            [s["precision"] for s in sweep],
            "-o",
            color=color,
            label=f"Epoch {r['epoch']}",
        )
        s = r["selected"]
        ax.scatter(s["recall"], s["precision"], s=160, marker="*", color=color)
        ax.annotate(
            f"t={s['threshold']:.2f}",
            (s["recall"], s["precision"]),
            xytext=(7, 7),
            textcoords="offset points",
        )
    ax.set(
        xlabel="Positive relation recall",
        ylabel="Positive relation precision",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Threshold sweep: tuning groups only",
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(out / "threshold_sweep.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
