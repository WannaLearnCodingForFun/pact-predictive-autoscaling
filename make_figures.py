"""Render paper figures from the CSVs written by ``pact.eval.export``.

Column names are exactly section 9.3. Missing or empty files raise; nothing
is invented to fill a gap.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from pact.eval.export import (
    ACTIONS_COLUMNS,
    DECISIONS_COLUMNS,
    FORECAST_COLUMNS,
    HORIZON_ERROR_COLUMNS,
    LATENCY_COLUMNS,
    REPLICAS_COLUMNS,
    TRADEOFF_COLUMNS,
    TRAINING_COLUMNS,
    UTILISATION_COLUMNS,
)

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

# Nine figures, one per 9.3 CSV. Names match the paper's figure roles.
FIGURES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("forecast.csv", FORECAST_COLUMNS, "fig3_forecast.pdf"),
    ("horizon_error.csv", HORIZON_ERROR_COLUMNS, "fig4_horizon_error.pdf"),
    ("replicas.csv", REPLICAS_COLUMNS, "fig5_replicas.pdf"),
    ("latency.csv", LATENCY_COLUMNS, "fig6_latency.pdf"),
    ("decisions.csv", DECISIONS_COLUMNS, "fig7_decisions.pdf"),
    ("tradeoff.csv", TRADEOFF_COLUMNS, "fig8_tradeoff.pdf"),
    ("actions.csv", ACTIONS_COLUMNS, "fig9_actions.pdf"),
    ("utilisation.csv", UTILISATION_COLUMNS, "fig2_utilisation.pdf"),
    ("training.csv", TRAINING_COLUMNS, "fig10_training.pdf"),
)


def _require_pandas_matplotlib() -> tuple[object, object]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "make_figures.py needs pandas and matplotlib; "
            'install with: pip install -e ".[viz]"'
        ) from exc
    return pd, plt


def load_csv(path: Path, columns: Sequence[str], pd: object) -> object:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing; run the experiment exporter before figures"
        )
    frame = pd.read_csv(path)  # type: ignore[union-attr]
    if frame.empty:
        raise ValueError(f"{path} has no computed rows")
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(
            f"{path} column names must be {list(columns)}; missing {missing}"
        )
    extra_ok = list(frame.columns[: len(columns)])
    if extra_ok != list(columns):
        raise ValueError(
            f"{path} leading columns must be {list(columns)}, got {extra_ok}"
        )
    return frame


def _save(fig: object, plt: object, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()  # type: ignore[union-attr]
    fig.savefig(dest)  # type: ignore[union-attr]
    plt.close(fig)  # type: ignore[union-attr]


def plot_forecast(frame: object, plt: object, dest: Path) -> None:
    fig, ax = plt.subplots()  # type: ignore[union-attr]
    ax.plot(frame["ts"], frame["actual"], label="actual")
    ax.plot(frame["ts"], frame["pred_h1"], label="pred_h1")
    ax.plot(frame["ts"], frame["pred_h12"], label="pred_h12")
    ax.set_xlabel("ts")
    ax.set_ylabel("utilisation")
    ax.legend()
    _save(fig, plt, dest)


def plot_horizon_error(frame: object, plt: object, dest: Path) -> None:
    fig, ax = plt.subplots()  # type: ignore[union-attr]
    ax.plot(frame["horizon"], frame["mae"], marker="o", label="mae")
    ax.plot(frame["horizon"], frame["rmse"], marker="o", label="rmse")
    ax.set_xlabel("horizon")
    ax.set_ylabel("error")
    ax.legend()
    _save(fig, plt, dest)


def plot_replicas(frame: object, plt: object, dest: Path) -> None:
    fig, ax = plt.subplots()  # type: ignore[union-attr]
    ax.plot(frame["ts"], frame["required"], label="required")
    for col in ("pact", "reactive", "arima", "lstm"):
        ax.plot(frame["ts"], frame[col], label=col)
    ax.set_xlabel("ts")
    ax.set_ylabel("replicas")
    ax.legend()
    _save(fig, plt, dest)


def plot_latency(frame: object, plt: object, dest: Path) -> None:
    fig, ax = plt.subplots()  # type: ignore[union-attr]
    for col in ("pact", "reactive", "arima", "lstm"):
        ax.plot(frame["ts"], frame[col], label=col)
    ax.plot(frame["ts"], frame["slo"], linestyle="--", label="slo")
    ax.set_xlabel("ts")
    ax.set_ylabel("latency")
    ax.legend()
    _save(fig, plt, dest)


def plot_decisions(frame: object, plt: object, dest: Path) -> None:
    fig, ax = plt.subplots()  # type: ignore[union-attr]
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = list(range(len(frame)))
    width = 0.2
    for i, metric in enumerate(metrics):
        offset = (i - 1.5) * width
        ax.bar([xi + offset for xi in x], frame[metric], width=width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(list(frame["method"]))
    ax.set_ylabel("score")
    ax.legend()
    _save(fig, plt, dest)


def plot_tradeoff(frame: object, plt: object, dest: Path) -> None:
    fig, ax = plt.subplots()  # type: ignore[union-attr]
    ax.scatter(frame["sla_violation"], frame["over_provision"])
    for _, row in frame.iterrows():  # type: ignore[union-attr]
        ax.annotate(str(row["method"]), (row["sla_violation"], row["over_provision"]))
    ax.set_xlabel("sla_violation")
    ax.set_ylabel("over_provision")
    _save(fig, plt, dest)


def plot_actions(frame: object, plt: object, dest: Path) -> None:
    fig, ax = plt.subplots()  # type: ignore[union-attr]
    for col in ("pact", "reactive", "arima", "lstm"):
        ax.plot(frame["ts"], frame[col], label=col)
    ax.set_xlabel("ts")
    ax.set_ylabel("action")
    ax.legend()
    _save(fig, plt, dest)


def plot_utilisation(frame: object, plt: object, dest: Path) -> None:
    fig, ax = plt.subplots()  # type: ignore[union-attr]
    methods = list(frame["method"].unique())
    data = [frame.loc[frame["method"] == m, "utilisation"] for m in methods]
    ax.boxplot(data, labels=methods)
    ax.set_ylabel("utilisation")
    _save(fig, plt, dest)


def plot_training(frame: object, plt: object, dest: Path) -> None:
    fig, ax = plt.subplots()  # type: ignore[union-attr]
    ax.plot(frame["epoch"], frame["train_loss"], label="train_loss")
    ax.plot(frame["epoch"], frame["val_loss"], label="val_loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    _save(fig, plt, dest)


PLOTTERS = {
    "forecast.csv": plot_forecast,
    "horizon_error.csv": plot_horizon_error,
    "replicas.csv": plot_replicas,
    "latency.csv": plot_latency,
    "decisions.csv": plot_decisions,
    "tradeoff.csv": plot_tradeoff,
    "actions.csv": plot_actions,
    "utilisation.csv": plot_utilisation,
    "training.csv": plot_training,
}


def render_all(results_dir: Path, figures_dir: Path) -> list[Path]:
    pd, plt = _require_pandas_matplotlib()
    written: list[Path] = []
    for filename, columns, figure_name in FIGURES:
        frame = load_csv(results_dir / filename, columns, pd)
        dest = figures_dir / figure_name
        PLOTTERS[filename](frame, plt, dest)
        written.append(dest)
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PACT figures from results CSVs")
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=FIGURES_DIR)
    args = parser.parse_args(argv)
    paths = render_all(args.results, args.out)
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
