"""Draw the segments exported by VideoProc in scores.txt."""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator


COLORS = {"D": "#087f70", "H": "#b75612", "V": "#3561ba"}
LABELS = {
    "D": "Diagonale  ·  Les deux vidéos avancent",
    "H": "Horizontale  ·  Vidéo 2 seule",
    "V": "Verticale  ·  Vidéo 1 seule",
}
INK = "#142a3b"
MUTED = "#506578"
GRID = "#e1e8ef"
PAGE = "#f6f8fb"
ROW = re.compile(
    r"^#\s+\d+\s*:\s*\[(\d+)\s*;\s*(\d+)\]\s*"
    r"==>\s*\[(\d+)\s*;\s*(\d+)\]",
    re.MULTILINE,
)


@dataclass(frozen=True)
class Segment:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def direction(self):
        if self.y0 == self.y1:
            return "H"
        if self.x0 == self.x1:
            return "V"
        return "D"


def read_segments(source):
    content = source.read_text(encoding="utf-8-sig")
    segments = []
    for row in ROW.finditer(content):
        y0, y1, x0, x1 = map(int, row.groups())
        dx, dy = x1 - x0, y1 - y0
        if dx < 0 or dy < 0 or (dx and dy and dx != dy):
            raise ValueError(f"Segment incompatible avec le chemin : {row.group()}")
        if dx or dy:
            segments.append(Segment(x0, y0, x1, y1))
    if not segments:
        raise ValueError("Le fichier ne contient aucun segment exploitable.")
    for previous, following in zip(segments, segments[1:]):
        if (previous.x1, previous.y1) != (following.x0, following.y0):
            raise ValueError("Deux segments successifs ne se rejoignent pas.")
    return segments


def zoom_bounds(segments):
    horizontal = [i for i, s in enumerate(segments) if s.direction == "H"]
    vertical = [i for i, s in enumerate(segments) if s.direction == "V"]
    if not horizontal or not vertical:
        return None
    pairs = [(min(h, v), max(h, v)) for h in horizontal for v in vertical]
    interior_pairs = [(first, last) for first, last in pairs if 0 < first and last < len(segments) - 1]
    first, last = min(
        interior_pairs or pairs,
        key=lambda p: max(
            segments[p[1]].x1 - segments[p[0]].x0,
            segments[p[1]].y1 - segments[p[0]].y0,
        ),
    )
    a, b = segments[first], segments[last]
    margin = max(16, round(max(b.x1 - a.x0, b.y1 - a.y0) * 0.42))
    return a.x0 - margin, a.y0 - margin, b.x1 + margin, b.y1 + margin


def draw_axis(ax, segments, bounds, *, zoom=False):
    x0, y0, x1, y1 = bounds
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axisbelow(True)
    ax.grid(color=GRID, linewidth=0.85)
    ax.set_xlabel("Images de la vidéo 2", color=INK, labelpad=11)
    ax.set_ylabel("Images de la vidéo 1", color=INK, labelpad=12)
    ax.tick_params(colors=MUTED, labelsize=10, length=3)
    format_tick = FuncFormatter(lambda value, _: f"{value:,.0f}".replace(",", "\u202f"))
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        axis.set_major_formatter(format_tick)
    for spine in ax.spines.values():
        spine.set_color("#c2ceda")
    for s in segments:
        ax.plot(
            (s.x0, s.x1),
            (s.y0, s.y1),
            color=COLORS[s.direction],
            linewidth=4.3 if zoom else 2.9,
            solid_capstyle="round",
        )
    if zoom:
        for s in segments:
            if s.direction == "D":
                continue
            if x0 <= s.x0 <= s.x1 <= x1 and y0 <= s.y0 <= s.y1 <= y1:
                ax.scatter(
                    [s.x0, s.x1],
                    [s.y0, s.y1],
                    s=25,
                    facecolors="white",
                    edgecolors=COLORS[s.direction],
                    linewidths=1.4,
                    zorder=5,
                )


def make_chart(segments, output):
    zoom = zoom_bounds(segments)
    fig = plt.figure(figsize=(14, 8), facecolor=PAGE)
    fig.text(0.07, 0.935, "Chemin d’alignement des deux vidéos", color=INK, fontsize=29, weight="bold")
    fig.text(
        0.071, 0.884,
        f"VideoProc / VSync  ·  {len(segments)} segments exportés par le programme",
        color=MUTED, fontsize=13,
    )
    if zoom:
        overview = fig.add_axes([0.075, 0.233, 0.365, 0.565], facecolor="white")
        detail = fig.add_axes([0.553, 0.233, 0.385, 0.565], facecolor="white")
    else:
        overview = fig.add_axes([0.22, 0.233, 0.56, 0.565], facecolor="white")
    start, end = segments[0], segments[-1]
    pad = max(2, round(max(end.x1 - start.x0, end.y1 - start.y0) * 0.035))
    draw_axis(
        overview, segments,
        (start.x0 - pad, start.y0 - pad, end.x1 + pad, end.y1 + pad),
    )
    overview.set_title("Vue d’ensemble", loc="left", color=INK, fontsize=15, pad=14, weight="bold")
    overview.scatter([start.x0, end.x1], [start.y0, end.y1], color=INK, s=24, zorder=5)
    if zoom:
        a, b, c, d = zoom
        center_x, center_y = (a + c) / 2, (b + d) / 2
        overview.scatter(
            [center_x], [center_y], s=125, facecolors="white",
            edgecolors=COLORS["V"], linewidths=2, zorder=6,
        )
        overview.annotate(
            "Zone agrandie", xy=(center_x, center_y), xytext=(29, 2), textcoords="offset points",
            ha="left", va="center", fontsize=10, color=MUTED,
            arrowprops={"arrowstyle": "-", "color": MUTED, "shrinkB": 6},
        )
        draw_axis(detail, segments, zoom, zoom=True)
        detail.set_title("Zoom sur les changements de direction", loc="left", color=INK, fontsize=15, pad=14, weight="bold")
    handles = [
        Line2D([0], [0], color=COLORS[direction], linewidth=4.2, label=LABELS[direction])
        for direction in ("D", "H", "V")
    ]
    fig.legend(
        handles=handles, loc="center", bbox_to_anchor=(0.5, 0.12),
        ncols=3, frameon=False, labelcolor=INK, fontsize=11,
        handlelength=1.8, columnspacing=1.9,
    )
    fig.text(
        0.071, 0.047,
        "Source  ·  Indices de début et de fin des segments dans le fichier texte de VideoProc.",
        color=MUTED, fontsize=10,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def make_frame_chart(segments, output, first, last):
    if not 1 <= first <= last <= len(segments):
        raise ValueError("L’extrait demandé est hors des segments de l’export.")
    excerpt = segments[first - 1:last]
    start, end = excerpt[0], excerpt[-1]
    span = max(end.x1 - start.x0, end.y1 - start.y0)
    padding = max(4, round(span * 0.032))
    bounds = (start.x0 - padding, start.y0 - padding, end.x1 + padding, end.y1 + padding)
    arrow_length = max(2, round(span * 0.024))
    point_size = min(14, 1200 / max(span, 1))

    fig = plt.figure(figsize=(12.5, 7.6), facecolor="white")
    fig.text(0.075, 0.927, "Chemin d’alignement, image par image",
             color=INK, fontsize=24, weight="bold")
    fig.text(0.077, 0.875, f"VideoProc / VSync  ·  Segments {first} à {last} de scores.txt",
             color=MUTED, fontsize=12)
    ax = fig.add_axes([0.095, 0.12, 0.60, 0.65], facecolor="#f9fbfd")
    draw_axis(ax, excerpt, bounds)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    for s in excerpt:
        dx, dy = s.x1 - s.x0, s.y1 - s.y0
        steps = max(dx, dy)
        step_x, step_y = dx // steps, dy // steps
        xs = [s.x0 + k * step_x for k in range(steps + 1)]
        ys = [s.y0 + k * step_y for k in range(steps + 1)]
        ax.scatter(xs, ys, color=COLORS[s.direction], s=point_size, linewidths=0, zorder=3)
        middle = (steps + 1) // 2
        if steps < arrow_length:
            continue
        ax.annotate(
            "",
            xy=(xs[middle], ys[middle]),
            xytext=(xs[max(0, middle - arrow_length)], ys[max(0, middle - arrow_length)]),
            arrowprops={
                "arrowstyle": "-|>",
                "color": COLORS[s.direction],
                "mutation_scale": 15,
                "lw": 2.4,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=5,
        )

    ax.scatter([start.x0, end.x1], [start.y0, end.y1],
               facecolors="white", edgecolors=INK, linewidths=1.7, s=45, zorder=6)
    handles = [
        Line2D([0], [0], color=COLORS[key], linewidth=3, marker="o", markersize=4, label=label)
        for key, label in (
            ("D", "Diagonale\nLes deux vidéos avancent"),
            ("H", "Droite\nLa vidéo 2 avance seule"),
            ("V", "Bas\nLa vidéo 1 avance seule"),
        )
    ]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.70, 0.53),
               frameon=False, labelcolor=INK, fontsize=11, labelspacing=2.0, handlelength=2)
    fig.text(0.077, 0.05, "Chaque point relie deux indices d’image du chemin exporté.",
             color=MUTED, fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def make_offset_chart(segments, output):
    offsets = [value for s in segments for value in (s.x0 - s.y0, s.x1 - s.y1)]
    first_frame, last_frame = segments[0].y0, segments[-1].y1
    x_pad = max(1, round((last_frame - first_frame) * 0.025))
    y_pad = max(4, round((max(offsets) - min(offsets)) * 0.18))

    fig = plt.figure(figsize=(12.5, 6.6), facecolor="white")
    fig.text(0.08, 0.91, "Comment le décalage évolue entre les vidéos",
             color=INK, fontsize=24, weight="bold")
    fig.text(0.082, 0.85, f"VideoProc / VSync  ·  {len(segments)} segments de scores.txt",
             color=MUTED, fontsize=12)
    ax = fig.add_axes([0.105, 0.235, 0.82, 0.52], facecolor="#f9fbfd")
    ax.set_xlim(first_frame - x_pad, last_frame + x_pad)
    ax.set_ylim(min(0, min(offsets)) - y_pad, max(0, max(offsets)) + y_pad)
    ax.set_xlabel("Indice d’image dans la vidéo 1", color=INK, labelpad=10)
    ax.set_ylabel("Décalage en images\nVidéo 2 − vidéo 1", color=INK, labelpad=12)
    ax.set_axisbelow(True)
    ax.grid(color=GRID, linewidth=0.85)
    ax.axhline(0, color="#8294a5", linewidth=1.15, linestyle=(0, (4, 4)))
    ax.tick_params(colors=MUTED, labelsize=10, length=3)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, prune="both"))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:,.0f}".replace(",", "\u202f"))
    )
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:+.0f}".replace("-", "−") if value else "0")
    )
    for spine in ax.spines.values():
        spine.set_color("#c2ceda")

    plateaus = {}
    for s in segments:
        offset_start, offset_end = s.x0 - s.y0, s.x1 - s.y1
        ax.plot(
            (s.y0, s.y1), (offset_start, offset_end),
            color=COLORS[s.direction], linewidth=3.3, solid_capstyle="round",
        )
        if s.direction == "D":
            existing = plateaus.get(offset_start)
            if existing is None or s.y1 - s.y0 > existing.y1 - existing.y0:
                plateaus[offset_start] = s

    longest = sorted(plateaus.items(), key=lambda item: item[1].y1 - item[1].y0, reverse=True)[:4]
    for offset, s in longest:
        signed = f"{offset:+d}".replace("-", "−") if offset else "0"
        ax.annotate(
            signed,
            xy=((s.y0 + s.y1) / 2, offset),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center", va="bottom",
            color=COLORS["D"], fontsize=12, weight="bold",
        )

    legend_labels = {
        "D": "Décalage constant",
        "H": "La vidéo 2 avance seule",
        "V": "La vidéo 1 avance seule",
    }
    handles = [
        Line2D([0], [0], color=COLORS[key], linewidth=3.3, label=label)
        for key, label in legend_labels.items()
    ]
    fig.legend(
        handles=handles, loc="center", bbox_to_anchor=(0.52, 0.11),
        ncols=3, frameon=False, labelcolor=INK, fontsize=10,
    )
    fig.text(0.082, 0.04, "Un décalage de +72 signifie que l’image associée est 72 indices plus loin dans la vidéo 2.",
             color=MUTED, fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def change_groups(segments, maximum_gap=90):
    groups = []
    for s in segments:
        if s.direction == "D":
            continue
        if not groups:
            groups.append([s])
            continue
        previous = groups[-1][-1]
        if max(s.x0 - previous.x1, s.y0 - previous.y1) > maximum_gap:
            groups.append([s])
        else:
            groups[-1].append(s)
    return groups


def show_video_axes(ax):
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Vidéo 2", color=INK, labelpad=6, fontsize=11)
    ax.set_ylabel("Vidéo 1", color=INK, labelpad=6, fontsize=11)
    ax.tick_params(labelsize=9)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MaxNLocator(nbins=4, integer=True))


def make_zoom_sheet(segments, output):
    groups = change_groups(segments)
    columns = min(4, len(groups) + 1)
    rows = (len(groups) + columns) // columns
    fig = plt.figure(figsize=(4.5 * columns, max(5.7, 5.05 * rows)), facecolor="white")
    fig.text(0.064, 0.934, "Chemin d’alignement et changements de direction",
             color=INK, fontsize=26, weight="bold")
    fig.text(
        0.065, 0.88, f"VideoProc / VSync  ·  {len(segments)} segments de scores.txt",
        color=MUTED, fontsize=13,
    )
    grid = fig.add_gridspec(
        rows, columns, left=0.065, right=0.972, top=0.765, bottom=0.185,
        wspace=0.42, hspace=0.92,
    )
    first, last = segments[0], segments[-1]
    pad = max(4, round(max(last.x1 - first.x0, last.y1 - first.y0) * 0.035))
    overview = fig.add_subplot(grid[0, 0], facecolor="#f9fbfd")
    draw_axis(
        overview, segments,
        (first.x0 - pad, first.y0 - pad, last.x1 + pad, last.y1 + pad),
    )
    show_video_axes(overview)
    overview.text(0.5, -0.13, "Chemin complet", ha="center", va="top",
                  transform=overview.transAxes, color=INK, fontsize=12, weight="bold")
    overview.text(0.5, -0.25, "Tous les segments de l’export", ha="center", va="top",
                  transform=overview.transAxes, color=MUTED, fontsize=10)

    for i, group in enumerate(groups, start=1):
        a, b = group[0], group[-1]
        span = max(b.x1 - a.x0, b.y1 - a.y0)
        half = max(6, (span + 1) // 2 + max(3, round(span * 0.17)))
        mid_x, mid_y = (a.x0 + b.x1) / 2, (a.y0 + b.y1) / 2
        ax = fig.add_subplot(grid[i // columns, i % columns], facecolor="#f9fbfd")
        left, top = max(-2, mid_x - half), max(-2, mid_y - half)
        draw_axis(ax, segments, (left, top, left + 2 * half, top + 2 * half), zoom=True)
        show_video_axes(ax)
        horizontal = sum(s.x1 - s.x0 for s in group if s.direction == "H")
        vertical = sum(s.y1 - s.y0 for s in group if s.direction == "V")
        changes = [f"Vidéo {video} seule +{amount}" for video, amount in ((2, horizontal), (1, vertical)) if amount]
        ax.text(0.5, -0.13, f"Zoom {i}", ha="center", va="top", transform=ax.transAxes,
                color=INK, fontsize=12, weight="bold")
        ax.text(0.5, -0.25, "  ·  ".join(changes), ha="center", va="top", transform=ax.transAxes,
                color=MUTED, fontsize=9.5)

    handles = [
        Line2D([0], [0], color=COLORS[key], linewidth=4, label=LABELS[key])
        for key in ("D", "H", "V")
    ]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.52, 0.054),
               ncols=3, frameon=False, labelcolor=INK, fontsize=11, columnspacing=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("graphique", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--segments", type=int, nargs=2, metavar=("DEBUT", "FIN"))
    mode.add_argument("--decalage", action="store_true")
    mode.add_argument("--zooms", action="store_true")
    args = parser.parse_args()
    segments = read_segments(args.scores)
    if args.zooms:
        make_zoom_sheet(segments, args.graphique)
    elif args.decalage:
        make_offset_chart(segments, args.graphique)
    elif args.segments:
        make_frame_chart(segments, args.graphique, *args.segments)
    else:
        make_chart(segments, args.graphique)
