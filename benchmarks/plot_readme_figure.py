"""Draw the README's headline figure from a saved admission sweep.

Fit time grows with the training table; a predict against the cached
context stays flat. Reads a `sweep_admission.py` baseline and writes a
light and a dark SVG (stdlib only, so it runs anywhere):

    python benchmarks/plot_readme_figure.py \
        benchmarks/baselines/sweep-a40x4-tabicl-v2.json docs/assets --features 100
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

W, H = 760, 380
LEFT, RIGHT, TOP, BOTTOM = 64, 180, 64, 52
X_MIN, X_MAX = 100, 100_000
Y_MIN, Y_MAX = 0.1, 200.0

# Validated with the dataviz palette checker (lightness band, CVD
# separation, contrast) against each theme's surface.
THEMES = {
    "light": {
        "surface": "#ffffff",
        "ink": "#1f2328",
        "muted": "#59636e",
        "grid": "#d1d9e0",
        "fit": "#6366F1",
        "predict": "#D97706",
    },
    "dark": {
        "surface": "#0d1117",
        "ink": "#f0f6fc",
        "muted": "#9198a1",
        "grid": "#30363d",
        "fit": "#7C83F0",
        "predict": "#C97A06",
    },
}
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"


def load_points(path: Path, n_features: int) -> tuple[str, list[dict]]:
    """The sweep's successful points at one width, inside the plotted row range."""
    data = json.loads(path.read_text())
    points = [
        p
        for p in data["points"]
        if p["n_features"] == n_features
        and p.get("status") == 200
        and "predict_s" in p
        and X_MIN <= p["n_rows"] <= X_MAX
    ]
    if not points:
        raise SystemExit(f"{path}: no successful {n_features}-feature points in range")
    return data["model"], sorted(points, key=lambda p: p["n_rows"])


def sx(rows: float) -> float:
    frac = (math.log10(rows) - math.log10(X_MIN)) / (
        math.log10(X_MAX) - math.log10(X_MIN)
    )
    return LEFT + frac * (W - LEFT - RIGHT)


def sy(seconds: float) -> float:
    frac = (math.log10(seconds) - math.log10(Y_MIN)) / (
        math.log10(Y_MAX) - math.log10(Y_MIN)
    )
    return H - BOTTOM - frac * (H - TOP - BOTTOM)


def fmt_s(seconds: float) -> str:
    return f"{seconds:.2f} s" if seconds < 1 else f"{seconds:.1f} s"


def render(model: str, n_features: int, points: list[dict], theme: dict) -> str:
    t = theme
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
        f'height="{H}" role="img" font-family="{FONT}" aria-label="Fit time grows with '
        f'the training table; predict time against the cached context stays flat">',
        f'<rect width="{W}" height="{H}" rx="12" fill="{t["surface"]}"/>',
        f'<text x="{LEFT}" y="30" font-size="17" font-weight="600" fill="{t["ink"]}">'
        "Fit once, then predict in a fraction of a second</text>",
        f'<text x="{LEFT}" y="50" font-size="13" fill="{t["muted"]}">{model}, {n_features} features, '
        "100-row predicts, one A40 GPU (log scales)</text>",
    ]
    for s in (0.1, 1, 10, 100):
        y = sy(s)
        out.append(
            f'<line x1="{LEFT}" x2="{W - RIGHT}" y1="{y:.1f}" y2="{y:.1f}" '
            f'stroke="{t["grid"]}" stroke-width="1"/>'
        )
        label = f"{s:g} s"
        out.append(
            f'<text x="{LEFT - 10}" y="{y + 4:.1f}" font-size="12" text-anchor="end" '
            f'fill="{t["muted"]}">{label}</text>'
        )
    for rows, label in (
        (100, "100"),
        (1_000, "1k"),
        (10_000, "10k"),
        (100_000, "100k"),
    ):
        out.append(
            f'<text x="{sx(rows):.1f}" y="{H - BOTTOM + 20}" font-size="12" '
            f'text-anchor="middle" fill="{t["muted"]}">{label}</text>'
        )
    out.append(
        f'<text x="{(LEFT + W - RIGHT) / 2:.1f}" y="{H - 12}" font-size="12" '
        f'text-anchor="middle" fill="{t["muted"]}">training rows</text>'
    )

    last = points[-1]
    for key, color, name in (
        ("fit_s", t["fit"], "fit: encode the table"),
        ("predict_s", t["predict"], "predict: cached context"),
    ):
        coords = " ".join(f"{sx(p['n_rows']):.1f},{sy(p[key]):.1f}" for p in points)
        out.append(
            f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5" '
            'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for p in points:
            out.append(
                f'<circle cx="{sx(p["n_rows"]):.1f}" cy="{sy(p[key]):.1f}" r="4" '
                f'fill="{color}" stroke="{t["surface"]}" stroke-width="2"/>'
            )
        x, y = sx(last["n_rows"]) + 12, sy(last[key])
        out.append(
            f'<text x="{x:.1f}" y="{y - 3:.1f}" font-size="13" font-weight="600" '
            f'fill="{t["ink"]}">{fmt_s(last[key])}</text>'
        )
        out.append(
            f'<text x="{x:.1f}" y="{y + 13:.1f}" font-size="12" fill="{t["muted"]}">{name}</text>'
        )

    # The gap at the largest table, as a bracket between the two lines.
    ratio = last["fit_s"] / last["predict_s"]
    bx, y1, y2 = (
        sx(last["n_rows"]) - 14,
        sy(last["fit_s"]) + 8,
        sy(last["predict_s"]) - 8,
    )
    out.append(
        f'<path d="M{bx + 5:.1f},{y1:.1f} H{bx:.1f} V{y2:.1f} H{bx + 5:.1f}" fill="none" '
        f'stroke="{t["muted"]}" stroke-width="1.5"/>'
    )
    out.append(
        # Low on the bracket, where it stays clear of the rising fit line.
        f'<text x="{bx - 8:.1f}" y="{sy(last["predict_s"] * 4):.1f}" font-size="15" font-weight="700" '
        f'text-anchor="end" fill="{t["ink"]}">{ratio:.0f}x faster</text>'
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the README's headline figure.")
    parser.add_argument("sweep", type=Path, help="a sweep_admission.py baseline JSON")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--features", type=int, default=100, help="table width to plot")
    args = parser.parse_args()
    model, points = load_points(args.sweep, args.features)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, theme in THEMES.items():
        path = args.out_dir / f"fit-vs-predict-{name}.svg"
        path.write_text(render(model, args.features, points, theme))
        print(path)


if __name__ == "__main__":
    main()
