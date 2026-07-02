"""
Lattice visualization for normalize_lattice_npz / standard_scale_lattice_npz output.

Legacy mode: histograms for lattice_normalize_summary.csv (scale_N / scale_V).

Std-variant compare mode (lattice_std_scaled_summary.csv):
  5 lattice variants × 5 plot types = 25 PNGs (default --compare-all).

Plot types:
  1. histogram   — distribution + mean ± σ, N stats on hover
  2. boxplot     — spread vs N bins
  3. violin      — full density per N bin
  4. density     — value vs N scatter + trend line
  5. ridge       — stacked KDE ridges per N bin

Requires: numpy, plotly, kaleido (PNG), scipy (options 4–5)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

DEFAULT_NORM_DIR = PIPELINE_DIR / "outputs" / "normalized_lattice"
AXIS_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c")
AXIS_NAMES = ("a", "b", "c")

LEGACY_METHODS = {
    "scale_N": {
        "title": "Normalized by N^(1/3) (CDVAE scale_length)",
        "keys": ("a_scaled_N", "b_scaled_N", "c_scaled_N"),
    },
    "scale_V": {
        "title": "Normalized by V^(1/3) (triclinic volume)",
        "keys": ("a_scaled_V", "b_scaled_V", "c_scaled_V"),
    },
}

STD_VARIANTS = {
    "var_V": {
        "title": "V scaling",
        "keys": ("a_var_V", "b_var_V", "c_var_V"),
        "slug": "v_scaling",
    },
    "var_N": {
        "title": "N scaling",
        "keys": ("a_var_N", "b_var_N", "c_var_N"),
        "slug": "n_scaling",
    },
    "var_std": {
        "title": "Standard scaling (raw a,b,c)",
        "keys": ("a_var_std", "b_var_std", "c_var_std"),
        "slug": "std_scaling",
    },
    "var_V_std": {
        "title": "V + Standard scaling",
        "keys": ("a_var_V_std", "b_var_V_std", "c_var_V_std"),
        "slug": "v_plus_std_scaling",
    },
    "var_N_std": {
        "title": "N + Standard scaling",
        "keys": ("a_var_N_std", "b_var_N_std", "c_var_N_std"),
        "slug": "n_plus_std_scaling",
    },
}

OPTION_SUFFIX = {
    "1": "opt1_histogram",
    "2": "opt2_boxplot",
    "3": "opt3_violin",
    "4": "opt4_density",
    "5": "opt5_ridge",
}


def parse_args() -> argparse.Namespace:
    default_norm = Path(os.environ.get("LATTICE_NORM_OUT_DIR", str(DEFAULT_NORM_DIR)))
    p = argparse.ArgumentParser(description="Lattice plots — legacy hist or 5×5 compare gallery.")
    p.add_argument("--norm-dir", type=Path, default=default_norm)
    p.add_argument("--records", type=Path, default=None)
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--bins", type=int, default=80, help="Histogram bins (option 1)")
    p.add_argument("--n-bins", type=int, default=8, help="N atom bins (options 2–5)")
    p.add_argument("--max-n-list", type=int, default=12)
    p.add_argument(
        "--format",
        choices=("png", "html", "both"),
        default="png",
    )
    p.add_argument(
        "--compare-all",
        action="store_true",
        default=False,
        help="Generate all 5 plot types × 5 variants (25 files) for std-scaled CSV",
    )
    p.add_argument(
        "--options",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated plot options: 1..5 (with --compare-all)",
    )
    return p.parse_args()


def load_records_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_records_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_rows(records_path: Path, csv_path: Path) -> list[dict[str, Any]]:
    if records_path.is_file():
        print(f"Loading: {records_path}")
        return load_records_jsonl(records_path)
    if csv_path.is_file():
        print(f"Loading: {csv_path}")
        return load_records_csv(csv_path)
    raise FileNotFoundError(f"No records found.\n  jsonl: {records_path}\n  csv:   {csv_path}")


def coerce_float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def _is_std_variant_csv(rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    return "a_var_V" in rows[0] and "a_var_N_std" in rows[0]


def _N_array(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([int(r["N"]) for r in rows], dtype=np.int64)


def _values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([coerce_float(r, key) for r in rows], dtype=np.float64)


def _n_bin_edges(N_vals: np.ndarray, n_bins: int) -> tuple[np.ndarray, list[str]]:
    lo, hi = int(N_vals.min()), int(N_vals.max())
    edges = np.linspace(lo, hi, n_bins + 1)
    labels = [f"{int(edges[i])}–{int(edges[i + 1])}" for i in range(n_bins)]
    return edges, labels


def _mask_for_bin(N_vals: np.ndarray, edges: np.ndarray, i: int) -> np.ndarray:
    lo, hi = edges[i], edges[i + 1]
    if i < len(edges) - 2:
        return (N_vals >= lo) & (N_vals < hi)
    return (N_vals >= lo) & (N_vals <= hi)


def build_binned_bar(
    values: np.ndarray,
    N: np.ndarray,
    *,
    label: str,
    n_bins: int,
    max_n_list: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    values = np.asarray(values, dtype=np.float64)
    N = np.asarray(N, dtype=np.int64)
    counts, edges = np.histogram(values, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hovers: list[str] = []
    for i in range(len(counts)):
        lo, hi = edges[i], edges[i + 1]
        mask = (values >= lo) & (values <= hi if i == len(counts) - 1 else values < hi)
        N_bin = N[mask]
        if len(N_bin) == 0:
            hovers.append(f"{label}<br>bin [{lo:.4f}, {hi:.4f}]<br>count: 0")
            continue
        N_sorted = np.sort(N_bin)
        sample = N_sorted[:max_n_list]
        more = len(N_sorted) - len(sample)
        sample_txt = ", ".join(str(int(x)) for x in sample)
        if more > 0:
            sample_txt += f", ... (+{more} more)"
        hovers.append(
            f"<b>{label}</b><br>"
            f"value bin: [{lo:.4f}, {hi:.4f}]<br>"
            f"<b>count: {int(counts[i])}</b><br>"
            f"N mean: {N_bin.mean():.1f}  median: {np.median(N_bin):.0f}<br>"
            f"N min–max: {int(N_bin.min())}–{int(N_bin.max())}<br>"
            f"N sample: {sample_txt}"
        )
    return centers, counts, hovers


# ---------------------------------------------------------------------------
# Option 1 — Histograms with mean ± σ and N hover
# ---------------------------------------------------------------------------


def make_option1_histogram(
    rows: list[dict[str, Any]],
    variant_key: str,
    *,
    n_bins: int,
    max_n_list: int,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    meta = STD_VARIANTS[variant_key]
    keys = meta["keys"]
    N = _N_array(rows)

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=[f"{ax} ({variant_key})" for ax in AXIS_NAMES],
        vertical_spacing=0.08,
    )

    for row_idx, (key, axis_name, color) in enumerate(zip(keys, AXIS_NAMES, AXIS_COLORS), 1):
        vals = _values(rows, key)
        centers, counts, hovers = build_binned_bar(
            vals, N, label=axis_name, n_bins=n_bins, max_n_list=max_n_list
        )
        width = (centers[1] - centers[0]) * 0.95 if len(centers) > 1 else 0.1
        fig.add_trace(
            go.Bar(
                x=centers,
                y=counts,
                width=width,
                marker_color=color,
                opacity=0.75,
                hovertext=hovers,
                hoverinfo="text",
                name=axis_name,
            ),
            row=row_idx,
            col=1,
        )
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="red",
            annotation_text=f"μ={mean_val:.3f}",
            row=row_idx,
            col=1,
        )
        fig.add_vrect(
            x0=mean_val - std_val,
            x1=mean_val + std_val,
            fillcolor="red",
            opacity=0.12,
            line_width=0,
            row=row_idx,
            col=1,
        )
        fig.update_xaxes(title_text=f"{axis_name} value", row=row_idx, col=1)
        fig.update_yaxes(title_text="Count of structures", row=row_idx, col=1)

    fig.update_layout(
        title=dict(
            text=(
                f"{meta['title']}<br>"
                f"<sup>n={len(rows)} structures | red dashed=mean, shaded band=±1σ</sup>"
            ),
            x=0.5,
        ),
        height=950,
        width=1000,
        showlegend=False,
        bargap=0.04,
    )
    return fig


# ---------------------------------------------------------------------------
# Option 2 — Box plots by N bins
# ---------------------------------------------------------------------------


def make_option2_boxplot(rows: list[dict[str, Any]], variant_key: str, *, n_bins: int):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    meta = STD_VARIANTS[variant_key]
    keys = meta["keys"]
    N_vals = _N_array(rows)
    edges, labels = _n_bin_edges(N_vals, n_bins)

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=[f"{ax} ({variant_key})" for ax in AXIS_NAMES],
        vertical_spacing=0.08,
    )

    for row_idx, (key, axis_name, color) in enumerate(zip(keys, AXIS_NAMES, AXIS_COLORS), 1):
        values = _values(rows, key)
        for i, label in enumerate(labels):
            mask = _mask_for_bin(N_vals, edges, i)
            subset = values[mask]
            if len(subset) == 0:
                continue
            fig.add_trace(
                go.Box(
                    y=subset,
                    x=[label] * len(subset),
                    name=label,
                    marker_color=color,
                    boxmean="sd",
                    hovertemplate=(
                        f"<b>{axis_name}</b><br>"
                        "N bin: %{x}<br>"
                        "value: %{y:.4f}<br>"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=row_idx,
                col=1,
            )
        fig.update_xaxes(title_text="N (atoms) bin", row=row_idx, col=1)
        fig.update_yaxes(title_text=f"{axis_name} value", row=row_idx, col=1)

    fig.update_layout(
        title=dict(
            text=f"{meta['title']} — Option 2: Box plots by N<br><sup>n={len(rows)}</sup>",
            x=0.5,
        ),
        height=1000,
        width=1100,
        showlegend=False,
        boxmode="group",
    )
    return fig


# ---------------------------------------------------------------------------
# Option 3 — Violin plots by N bins
# ---------------------------------------------------------------------------


def make_option3_violin(rows: list[dict[str, Any]], variant_key: str, *, n_bins: int):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    meta = STD_VARIANTS[variant_key]
    keys = meta["keys"]
    N_vals = _N_array(rows)
    edges, labels = _n_bin_edges(N_vals, n_bins)

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=[f"{ax} ({variant_key})" for ax in AXIS_NAMES],
        vertical_spacing=0.08,
    )

    for row_idx, (key, axis_name, color) in enumerate(zip(keys, AXIS_NAMES, AXIS_COLORS), 1):
        values = _values(rows, key)
        for i, label in enumerate(labels):
            mask = _mask_for_bin(N_vals, edges, i)
            subset = values[mask]
            if len(subset) < 3:
                continue
            fig.add_trace(
                go.Violin(
                    y=subset,
                    x=[label] * len(subset),
                    name=label,
                    marker_color=color,
                    opacity=0.65,
                    box_visible=True,
                    meanline_visible=True,
                    points=False,
                    hovertemplate=(
                        f"<b>{axis_name}</b><br>"
                        "N bin: %{x}<br>"
                        "value: %{y:.4f}<br>"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=row_idx,
                col=1,
            )
        fig.update_xaxes(title_text="N (atoms) bin", row=row_idx, col=1)
        fig.update_yaxes(title_text=f"{axis_name} value", row=row_idx, col=1)

    fig.update_layout(
        title=dict(
            text=f"{meta['title']} — Option 3: Violin plots by N<br><sup>n={len(rows)}</sup>",
            x=0.5,
        ),
        height=1000,
        width=1100,
        showlegend=False,
        violinmode="group",
    )
    return fig


# ---------------------------------------------------------------------------
# Option 4 — Density scatter: value vs N + trend
# ---------------------------------------------------------------------------


def make_option4_density(rows: list[dict[str, Any]], variant_key: str, *, seed: int = 42):
    from scipy.ndimage import uniform_filter1d

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    meta = STD_VARIANTS[variant_key]
    keys = meta["keys"]
    N_vals = _N_array(rows)
    rng = np.random.default_rng(seed)

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=[f"{ax} ({variant_key})" for ax in AXIS_NAMES],
        vertical_spacing=0.08,
    )

    for row_idx, (key, axis_name, color) in enumerate(zip(keys, AXIS_NAMES, AXIS_COLORS), 1):
        values = _values(rows, key)
        if len(values) > 5000:
            idx = rng.choice(len(values), 5000, replace=False)
            N_s = N_vals[idx]
            v_s = values[idx]
        else:
            N_s, v_s = N_vals, values

        fig.add_trace(
            go.Scatter(
                x=N_s,
                y=v_s,
                mode="markers",
                marker=dict(
                    size=4,
                    color=v_s,
                    colorscale="Viridis",
                    opacity=0.45,
                    showscale=row_idx == 1,
                    colorbar=dict(title=f"{axis_name}") if row_idx == 1 else None,
                ),
                name=axis_name,
                hovertemplate=(
                    f"<b>{axis_name}</b><br>"
                    "N: %{x}<br>"
                    "value: %{y:.4f}<br>"
                    "<extra></extra>"
                ),
            ),
            row=row_idx,
            col=1,
        )

        order = np.argsort(N_s)
        N_sorted = N_s[order]
        v_sorted = v_s[order]
        window = max(50, len(v_sorted) // 50)
        v_smooth = uniform_filter1d(v_sorted.astype(np.float64), size=window)
        fig.add_trace(
            go.Scatter(
                x=N_sorted,
                y=v_smooth,
                mode="lines",
                line=dict(color="red", width=2),
                name="trend",
                hoverinfo="skip",
            ),
            row=row_idx,
            col=1,
        )
        fig.update_xaxes(title_text="N (atoms)", row=row_idx, col=1)
        fig.update_yaxes(title_text=f"{axis_name} value", row=row_idx, col=1)

    fig.update_layout(
        title=dict(
            text=(
                f"{meta['title']} — Option 4: Density scatter vs N<br>"
                f"<sup>n={len(rows)} | red=trend | subsample≤5000 for clarity</sup>"
            ),
            x=0.5,
        ),
        height=1000,
        width=1100,
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Option 5 — Ridge plots (KDE per N bin, one panel per axis)
# ---------------------------------------------------------------------------


def make_option5_ridge(rows: list[dict[str, Any]], variant_key: str, *, n_bins: int = 6):
    from scipy.stats import gaussian_kde

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    meta = STD_VARIANTS[variant_key]
    keys = meta["keys"]
    N_vals = _N_array(rows)
    edges, _ = _n_bin_edges(N_vals, n_bins)

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=[f"{ax} ridge ({variant_key})" for ax in AXIS_NAMES],
        vertical_spacing=0.10,
    )

    for row_idx, (key, axis_name, color) in enumerate(zip(keys, AXIS_NAMES, AXIS_COLORS), 1):
        values = _values(rows, key)
        ridge_idx = 0
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            mask = _mask_for_bin(N_vals, edges, i)
            subset = values[mask]
            if len(subset) < 10:
                continue
            kde = gaussian_kde(subset)
            pad = max((subset.max() - subset.min()) * 0.15, 1e-6)
            x_range = np.linspace(subset.min() - pad, subset.max() + pad, 200)
            density = kde(x_range)
            density = density / (density.max() + 1e-12)
            offset = ridge_idx * 0.35
            ridge_idx += 1
            label = f"N {int(lo)}–{int(hi)}"
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=density + offset,
                    fill="tozeroy",
                    mode="lines",
                    line=dict(color=color, width=1.2),
                    fillcolor=color,
                    opacity=0.35,
                    name=label,
                    hovertemplate=(
                        f"<b>{axis_name}</b> {label}<br>"
                        "value: %{x:.4f}<br>"
                        "rel. density: %{y:.3f}<br>"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=row_idx,
                col=1,
            )

        fig.update_yaxes(showticklabels=False, title_text="", row=row_idx, col=1)
        fig.update_xaxes(title_text=f"{axis_name} value", row=row_idx, col=1)

    fig.update_layout(
        title=dict(
            text=f"{meta['title']} — Option 5: Ridge plots by N<br><sup>n={len(rows)}</sup>",
            x=0.5,
        ),
        height=1000,
        width=1000,
        showlegend=False,
    )
    return fig


OPTION_MAKERS: dict[str, Callable[..., Any]] = {
    "1": make_option1_histogram,
    "2": make_option2_boxplot,
    "3": make_option3_violin,
    "4": make_option4_density,
    "5": make_option5_ridge,
}


def make_legacy_method_figure(
    rows: list[dict[str, Any]],
    method_key: str,
    *,
    n_bins: int,
    max_n_list: int,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    meta = LEGACY_METHODS[method_key]
    ak, bk, ck = meta["keys"]
    N = _N_array(rows)

    panels = [
        ("N", np.array([int(r["N"]) for r in rows], dtype=np.float64), "Number of atoms", "atoms"),
        ("a", _values(rows, ak), f"a' ({method_key})", "Å"),
        ("b", _values(rows, bk), f"b' ({method_key})", "Å"),
        ("c", _values(rows, ck), f"c' ({method_key})", "Å"),
    ]

    fig = make_subplots(rows=2, cols=2, subplot_titles=[p[2] for p in panels], vertical_spacing=0.10)

    for (name, vals, title, unit), (row, col) in zip(
        panels, [(1, 1), (1, 2), (2, 1), (2, 2)]
    ):
        centers, counts, hovers = build_binned_bar(vals, N, label=title, n_bins=n_bins, max_n_list=max_n_list)
        fig.add_trace(
            go.Bar(
                x=centers,
                y=counts,
                hovertext=hovers,
                hoverinfo="text",
                marker_color="steelblue" if name != "N" else "darkorange",
                width=(centers[1] - centers[0]) * 0.9 if len(centers) > 1 else 0.1,
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text=f"{title} ({unit})", row=row, col=col)
        fig.update_yaxes(title_text="Count", row=row, col=col)

    fig.update_layout(
        title=dict(text=f"{meta['title']}<br><sup>n={len(rows)}</sup>", x=0.5),
        showlegend=False,
        height=840,
        width=1200,
    )
    return fig


def _save_figure(fig, base_path: Path, fmt: str, *, height: int = 1000, width: int = 1100) -> list[Path]:
    written: list[Path] = []
    if fmt in ("html", "both"):
        out_html = base_path.with_suffix(".html")
        fig.write_html(str(out_html), include_plotlyjs="cdn")
        written.append(out_html)
    if fmt in ("png", "both"):
        out_png = base_path.with_suffix(".png")
        try:
            fig.write_image(str(out_png), width=width, height=height, scale=2)
        except Exception as exc:
            raise ImportError("PNG export failed. pip install kaleido") from exc
        written.append(out_png)
    return written


def _parse_option_ids(options_str: str) -> list[str]:
    ids = [s.strip() for s in options_str.split(",") if s.strip()]
    for oid in ids:
        if oid not in OPTION_MAKERS:
            raise ValueError(f"Unknown option {oid!r}. Use 1,2,3,4,5")
    return ids


def run_compare_gallery(
    rows: list[dict[str, Any]],
    plot_dir: Path,
    *,
    option_ids: list[str],
    fmt: str,
    hist_bins: int,
    n_bins: int,
    max_n_list: int,
) -> list[Path]:
    try:
        import plotly  # noqa: F401
        from scipy.ndimage import uniform_filter1d  # noqa: F401
        from scipy.stats import gaussian_kde  # noqa: F401
    except ImportError as exc:
        raise ImportError("Need: pip install plotly scipy kaleido") from exc

    compare_dir = plot_dir / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for vkey, meta in STD_VARIANTS.items():
        slug = meta["slug"]
        for oid in option_ids:
            suffix = OPTION_SUFFIX[oid]

            if oid == "1":
                fig = make_option1_histogram(rows, vkey, n_bins=hist_bins, max_n_list=max_n_list)
            elif oid == "2":
                fig = make_option2_boxplot(rows, vkey, n_bins=n_bins)
            elif oid == "3":
                fig = make_option3_violin(rows, vkey, n_bins=n_bins)
            elif oid == "4":
                fig = make_option4_density(rows, vkey)
            else:
                fig = make_option5_ridge(rows, vkey, n_bins=min(n_bins, 6))

            base = compare_dir / f"{slug}_{suffix}"
            paths = _save_figure(fig, base, fmt)
            written.extend(paths)
            for p in paths:
                print(f"Wrote: {p}")

    return written


def run_plots(
    *,
    norm_dir: Path,
    out_dir: Path | None = None,
    records_path: Path | None = None,
    csv_path: Path | None = None,
    n_bins: int = 80,
    n_atom_bins: int = 8,
    max_n_list: int = 12,
    fmt: str = "png",
    compare_all: bool = False,
    option_ids: list[str] | None = None,
) -> list[Path]:
    norm_dir = norm_dir.resolve()
    records_path = records_path or (norm_dir / "lattice_normalize_records.jsonl")
    csv_path = csv_path or (norm_dir / "lattice_normalize_summary.csv")
    plot_dir = (out_dir or (norm_dir / "plots")).resolve()

    rows = load_rows(records_path, csv_path)
    if not rows:
        raise ValueError("No records in input file.")

    if _is_std_variant_csv(rows) and compare_all:
        oids = option_ids or ["1", "2", "3", "4", "5"]
        print(f"Compare gallery: {len(STD_VARIANTS)} variants × {len(oids)} plot types")
        return run_compare_gallery(
            rows,
            plot_dir,
            option_ids=oids,
            fmt=fmt,
            hist_bins=n_bins,
            n_bins=n_atom_bins,
            max_n_list=max_n_list,
        )

    try:
        import plotly  # noqa: F401
    except ImportError as exc:
        raise ImportError("plotly required: pip install plotly") from exc

    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if _is_std_variant_csv(rows):
        oids = option_ids or ["1"]
        print(f"Std-variant plots (use --compare-all for all 25): option {oids[0]}")
        return run_compare_gallery(
            rows,
            plot_dir,
            option_ids=oids,
            fmt=fmt,
            hist_bins=n_bins,
            n_bins=n_atom_bins,
            max_n_list=max_n_list,
        )

    print("Legacy lattice_normalize format.")
    for method_key in LEGACY_METHODS:
        fig = make_legacy_method_figure(rows, method_key, n_bins=n_bins, max_n_list=max_n_list)
        paths = _save_figure(fig, plot_dir / f"lattice_hist_{method_key}", fmt)
        written.extend(paths)
        for p in paths:
            print(f"Wrote: {p}")
    return written


def main() -> None:
    args = parse_args()
    norm_dir = args.norm_dir.resolve()
    records_path = args.records or (norm_dir / "lattice_normalize_records.jsonl")
    csv_path = args.csv or (norm_dir / "lattice_normalize_summary.csv")
    out_dir = (args.out_dir or (norm_dir / "plots")).resolve()

    # Auto-enable compare-all for std-scaled summary CSV
    compare_all = args.compare_all
    if not compare_all and csv_path.is_file():
        try:
            with csv_path.open(encoding="utf-8", newline="") as fh:
                header = fh.readline()
            if "a_var_V" in header and "a_var_N_std" in header:
                compare_all = True
        except OSError:
            pass

    try:
        option_ids = _parse_option_ids(args.options)
        written = run_plots(
            norm_dir=norm_dir,
            out_dir=out_dir,
            records_path=records_path,
            csv_path=csv_path,
            n_bins=args.bins,
            n_atom_bins=args.n_bins,
            max_n_list=args.max_n_list,
            fmt=args.format,
            compare_all=compare_all,
            option_ids=option_ids,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        sys.exit(1)
    except ImportError as exc:
        print(exc)
        sys.exit(1)

    print()
    print(f"Done — {len(written)} file(s), format={args.format}")
    if compare_all:
        print(f"Gallery folder: {out_dir / 'compare'}")


if __name__ == "__main__":
    main()
