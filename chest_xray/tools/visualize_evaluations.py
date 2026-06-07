from typing import Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap


# ── LaTeX helpers ─────────────────────────────────────────────────────────────

def summary_to_latex(summary: Dict[str, Any], label: str = "tab:summary") -> str:
    display_names = {
        "Macro AUROC": "Macro AUROC",
        "Macro Precision": "Macro Precision",
        "Macro Recall": "Macro Recall",
        "Macro F1-score": "Macro F1-Score",
        "Inference Time Seconds": "Inference Time (s)",
        "Inference Time Per Image Seconds": "Inference Time per Image (s)",
        "Total Parameters": "Total Parameters",
        "Trainable Parameters": "Trainable Parameters",
    }
    rows = []
    for key, value in summary.items():
        name = display_names.get(key, key)
        if isinstance(value, float):
            # Integer-valued floats (unlikely here, but safe).
            formatted = f"{value:.6f}".rstrip("0").rstrip(".")
        elif isinstance(value, int):
            formatted = f"{value:,}"
        else:
            formatted = str(value)
        rows.append(f"  {name} & {formatted} \\\\")
    body = "\n".join(rows)
    return (
        "\\begin{table}[ht]\n"
        "  \\centering\n"
        "  \\caption{Model Evaluation Summary}\n"
        f"  \\label{{{label}}}\n"
        "  \\begin{tabular}{lr}\n"
        "    \\toprule\n"
        "    \\textbf{Metric} & \\textbf{Value} \\\\\n"
        "    \\midrule\n"
        f"{body}\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}"
    )


def results_df_to_latex(
    results_df: pd.DataFrame,
    label: str = "tab:per_disease",
    float_fmt: str = "{:.4f}",
) -> str:
    metric_cols = ["AUROC", "Precision", "Recall", "F1-score"]
    df = results_df[["Disease"] + metric_cols].copy()
    col_headers = {
        "Disease": "\\textbf{Disease}",
        "AUROC": "\\textbf{AUROC}",
        "Precision": "\\textbf{Precision}",
        "Recall": "\\textbf{Recall}",
        "F1-score": "\\textbf{F1-Score}",
    }
    header = " & ".join(col_headers[c] for c in df.columns) + " \\\\"
    rows = []
    for _, row in df.iterrows():
        cells = [str(row["Disease"])]
        for col in metric_cols:
            val = row[col]
            cells.append(float_fmt.format(val) if not np.isnan(val) else "---")
        rows.append("  " + " & ".join(cells) + " \\\\")
    macro_cells = ["\\textbf{Macro Average}"]
    for col in metric_cols:
        macro_cells.append(float_fmt.format(results_df[col].mean()))
    rows.append("  \\midrule")
    rows.append("  " + " & ".join(macro_cells) + " \\\\")
    body = "\n".join(rows)
    n_cols = len(df.columns)
    col_spec = "l" + "r" * (n_cols - 1)
    return (
        "\\begin{table}[ht]\n"
        "  \\centering\n"
        "  \\caption{Per-Disease Classification Metrics}\n"
        f"  \\label{{{label}}}\n"
        f"  \\begin{{tabular}}{{{col_spec}}}\n"
        "    \\toprule\n"
        f"    {header}\n"
        "    \\midrule\n"
        f"{body}\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}"
    )



def plot_confusion_matrices(
    confusion_matrices: Dict[str, Dict],
    save_path: str | Path | None = None,
    ncols: int = 4,
    normalize: bool = True,
    figsize_per_cell: tuple[float, float] = (2.6, 2.8),
) -> plt.Figure:
    diseases = list(confusion_matrices.keys())
    n = len(diseases)
    nrows = int(np.ceil(n / ncols))
    fig_w = figsize_per_cell[0] * ncols
    fig_h = figsize_per_cell[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), constrained_layout=False)
    axes = np.array(axes).reshape(-1)  # flatten for easy iteration
    cmap = LinearSegmentedColormap.from_list(
        "cm_blue", ["#ffffff", "#1a4e8a"], N=256
    )
    cell_labels = ["TN", "FP", "FN", "TP"]
    for ax, disease in zip(axes, diseases):
        cm_data = confusion_matrices[disease]
        cm = cm_data["matrix"].astype(float)  # shape (2, 2)
        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            # Avoid divide-by-zero for classes absent from the split.
            cm_display = np.where(row_sums > 0, cm / row_sums, 0.0)
        else:
            cm_display = cm
        im = ax.imshow(cm_display, cmap=cmap, vmin=0, vmax=1 if normalize else None)
        for flat_idx, label in enumerate(cell_labels):
            r, c = divmod(flat_idx, 2)
            raw = int(cm[r, c])
            rate = cm_display[r, c]
            text_color = "white" if rate > 0.55 else "#1a1a2e"

            if normalize:
                ax.text(
                    c, r,
                    f"{rate:.2f}\n({raw})",
                    ha="center", va="center",
                    fontsize=9, color=text_color,
                    fontweight="bold",
                )
            else:
                ax.text(
                    c, r,
                    f"{raw}",
                    ha="center", va="center",
                    fontsize=10, color=text_color,
                    fontweight="bold",
                )
        ax.set_title(disease, fontsize=10, fontweight="bold", pad=6)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Neg", "Pred Pos"], fontsize=8)
        ax.set_yticklabels(["Actual Neg", "Actual Pos"], fontsize=8)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in axes[n:]:
        ax.set_visible(False)
    if normalize:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Row-Normalised Rate", fontsize=9)
        cbar.ax.tick_params(labelsize=8)
    norm_note = " (normalised; raw counts in parentheses)" if normalize else ""
    fig.suptitle(
        f"Per-Disease Confusion Matrices{norm_note}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.subplots_adjust(
        left=0.06, right=0.90 if normalize else 0.98,
        top=0.93, bottom=0.05,
        hspace=0.45, wspace=0.35,
    )
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig



def visualize_evals(
    results_df: pd.DataFrame,
    summary: Dict[str, Any],
    confusion_matrices: Dict[str, Dict],
    output_dir: str | Path = ".",
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_tex = summary_to_latex(summary)
    (out / "summary_table.tex").write_text(summary_tex)
    print(f"Wrote {out / 'summary_table.tex'}")
    disease_tex = results_df_to_latex(results_df)
    (out / "per_disease_table.tex").write_text(disease_tex)
    print(f"Wrote {out / 'per_disease_table.tex'}")
    fig = plot_confusion_matrices(
        confusion_matrices,
        save_path=out / "confusion_matrices.png",
    )
    plt.close(fig)
    print(f"Wrote {out / 'confusion_matrices.png'}")