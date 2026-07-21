"""Render manuscript-native PNG figures from audited derived CSV files.

The vector SVGs remain the editable sources.  This Pillow renderer produces the
high-resolution PNG assets consumed by pdfLaTeX/Overleaf without browser-specific
SVG behavior.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "paper" / "data" / "derived"
FIGURES = ROOT / "paper" / "manuscript" / "figures"
SCALE = 2


def rows(name: str) -> list[dict[str, str]]:
    with (DERIVED / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = [
        "arialbd.ttf" if bold else "arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    search_roots = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu")]
    for root in search_roots:
        for name in names:
            path = root / name
            if path.is_file():
                return ImageFont.truetype(str(path), size * SCALE)
    return ImageFont.load_default(size=size * SCALE)


def canvas(width: int, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width * SCALE, height * SCALE), "white")
    return image, ImageDraw.Draw(image)


def box(draw: ImageDraw.ImageDraw, xy: Iterable[float], **kwargs: object) -> None:
    draw.rounded_rectangle(tuple(int(value * SCALE) for value in xy), **kwargs)


def centered(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    value: str,
    size: int,
    *,
    bold: bool = False,
    fill: str = "#172033",
) -> None:
    draw.text(
        (int(xy[0] * SCALE), int(xy[1] * SCALE)),
        value,
        font=font(size, bold=bold),
        fill=fill,
        anchor="mm",
    )


def right_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    value: str,
    size: int,
    *,
    bold: bool = False,
    fill: str = "#172033",
) -> None:
    draw.text(
        (int(xy[0] * SCALE), int(xy[1] * SCALE)),
        value,
        font=font(size, bold=bold),
        fill=fill,
        anchor="rm",
    )


def save(image: Image.Image, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    image.save(FIGURES / name, format="PNG", optimize=True, dpi=(220, 220))


def render_readiness() -> None:
    width, height = 1800, 650
    image, draw = canvas(width, height)
    statuses = rows("routability_gates.csv")
    titles = ["1. Interface", "2. Capability", "3. Complementarity", "4. Cost", "5. Selectability"]
    lines = [
        ("3 eligible routes", "0% parseable"),
        ("Oracle ceiling 12.5%", "required floor 80%"),
        ("Best fixed = oracle", "headroom 0.0 pp"),
        ("1,320 calls sampled", "generation only; 4 boards"),
        ("0 of 9 thresholds", "validation-feasible"),
    ]
    status_fill = {"FAIL": "#fde8e8", "PARTIAL": "#fff4d6", "QUALIFIED": "#e7f0fb"}
    status_stroke = {"FAIL": "#b4232f", "PARTIAL": "#9a6700", "QUALIFIED": "#245f9e"}
    centered(draw, (width / 2, 50), "Failure-first audit of the frozen resident-eligible pool", 32, bold=True)
    centered(draw, (width / 2, 88), "Each question is a prerequisite for defensible constrained routing", 20, fill="#536173")
    box_w, box_h, gap, start_x, y = 300, 320, 45, 38, 135
    for index, record in enumerate(statuses):
        status = record["status"]
        x = start_x + index * (box_w + gap)
        stroke = status_stroke[status]
        box(
            draw,
            (x, y, x + box_w, y + box_h),
            radius=22 * SCALE,
            fill=status_fill[status],
            outline=stroke,
            width=4 * SCALE,
        )
        centered(draw, (x + box_w / 2, y + 55), titles[index], 24, bold=True)
        box(
            draw,
            (x + 80, y + 88, x + box_w - 80, y + 136),
            radius=16 * SCALE,
            fill=stroke,
        )
        centered(draw, (x + box_w / 2, y + 112), status, 20, bold=True, fill="white")
        centered(draw, (x + box_w / 2, y + 202), lines[index][0], 20, bold=True)
        centered(draw, (x + box_w / 2, y + 246), lines[index][1], 20)
        if index < len(statuses) - 1:
            x1, x2, mid = x + box_w + 7, x + box_w + gap, y + box_h / 2
            draw.line((x1 * SCALE, mid * SCALE, (x2 - 18) * SCALE, mid * SCALE), fill="#6d7888", width=5 * SCALE)
            draw.polygon(
                [((x2 - 18) * SCALE, (mid - 11) * SCALE), ((x2 - 18) * SCALE, (mid + 11) * SCALE), (x2 * SCALE, mid * SCALE)],
                fill="#6d7888",
            )
    box(draw, (360, 515, 1440, 587), radius=18 * SCALE, fill="#eef2f7", outline="#50627a", width=2 * SCALE)
    centered(draw, (900, 551), "Audit outcome: redesign the eligible pool before interpreting router optimization.", 24, bold=True)
    save(image, "routability_audit.png")


def render_headroom() -> None:
    width, height = 1700, 900
    image, draw = canvas(width, height)
    data = {row["pool_id"]: row for row in rows("routability_pools.csv")}
    order = ["deployable", "granite_pair", "peer_pair", "upper_pair", "offline_reference", "all_retained"]
    labels = {
        "deployable": ("Resident eligible", "11 actions"),
        "granite_pair": ("Granite 3B", "direct / grounded"),
        "peer_pair": ("SmolLM3 3B", "direct / grounded"),
        "upper_pair": ("Qwen3 4B", "direct / grounded"),
        "offline_reference": ("References", "6 actions"),
        "all_retained": ("Complete", "17 actions"),
    }
    left, right, top, bottom, maximum = 125, 55, 105, 175, 0.8
    plot_h = height - top - bottom

    def sy(value: float) -> float:
        return top + plot_h * (1 - value / maximum)

    centered(draw, (width / 2, 47), "Routing opportunity is a property of the action pool", 32, bold=True)
    centered(draw, (width / 2, 80), "Best fixed utility versus a post-hoc per-query oracle", 20, fill="#536173")
    for tick in range(0, 81, 10):
        y = sy(tick / 100)
        draw.line((left * SCALE, y * SCALE, (width - right) * SCALE, y * SCALE), fill="#e1e5eb", width=2 * SCALE)
        right_text(draw, (left - 16, y), str(tick), 18)
    group_w, bar_w = (width - left - right) / len(order), 72
    for index, pool_id in enumerate(order):
        record = data[pool_id]
        center = left + group_w * (index + 0.5)
        best, oracle = float(record["best_fixed_success"]), float(record["oracle_success"])
        for offset, value, color in [(-bar_w / 2, best, "#7f8c9d"), (bar_w / 2, oracle, "#2563a6")]:
            x, y = center + offset - bar_w / 2, sy(value)
            box(draw, (x, y, x + bar_w, top + plot_h), radius=5 * SCALE, fill=color)
            centered(draw, (x + bar_w / 2, y - 14), f"{100 * value:.1f}", 18, bold=True)
        centered(draw, (center, top + plot_h + 38), labels[pool_id][0], 20, bold=True)
        centered(draw, (center, top + plot_h + 67), labels[pool_id][1], 17, fill="#536173")
        centered(draw, (center, top + plot_h + 109), f"H = {100 * float(record['routing_headroom']):.1f} pp", 18, bold=True, fill="#1f4f82")
    draw.multiline_text(
        (58 * SCALE, (top + plot_h / 2) * SCALE),
        "Strict\nsuccess\n(%)",
        font=font(18, bold=True),
        fill="#172033",
        anchor="mm",
        align="center",
        spacing=4 * SCALE,
    )
    draw.rectangle((610 * SCALE, 852 * SCALE, 636 * SCALE, 870 * SCALE), fill="#7f8c9d")
    draw.text((650 * SCALE, 861 * SCALE), "Best fixed", font=font(18), fill="#172033", anchor="lm")
    draw.rectangle((820 * SCALE, 852 * SCALE, 846 * SCALE, 870 * SCALE), fill="#2563a6")
    draw.text((860 * SCALE, 861 * SCALE), "Per-query oracle", font=font(18), fill="#172033", anchor="lm")
    save(image, "action_pool_headroom.png")


def render_heatmap() -> None:
    width, height = 1820, 800
    image, draw = canvas(width, height)
    data = {(row["model_key"], row["task_type"]): float(row["grounded_minus_direct"]) for row in rows("task_grounding_effects.csv")}
    models = ["tiny", "small", "granite", "peer", "upper"]
    model_labels = {
        "tiny": "Qwen3 0.6B",
        "small": "Granite 4.0 1B",
        "granite": "Granite 4.1 3B",
        "peer": "SmolLM3 3B",
        "upper": "Qwen3 4B Instruct",
    }
    tasks = ["no_retrieval", "knowledge_only", "memory_only", "knowledge_memory", "temporal_update", "authority_conflict", "multi_hop", "deleted_or_missing"]
    task_labels = ["Direct copy", "Knowledge", "Memory", "Know. + mem.", "Temporal", "Authority", "Two hop", "Deleted"]

    def color(value: float) -> str:
        neutral = (245, 247, 250)
        target = (34, 102, 172) if value >= 0 else (178, 32, 48)
        intensity = min(1.0, abs(value))
        rgb = tuple(round(neutral[i] + intensity * (target[i] - neutral[i])) for i in range(3))
        return "#" + "".join(f"{component:02x}" for component in rgb)

    centered(draw, (width / 2, 47), "Grounding effects reverse across generators and task classes", 32, bold=True)
    centered(draw, (width / 2, 82), "Grounded minus direct strict success (percentage points; 15 questions per cell)", 20, fill="#536173")
    left, top, cell_w, cell_h = 305, 155, 178, 92
    for index, label in enumerate(task_labels):
        centered(draw, (left + index * cell_w + cell_w / 2, top - 28), label, 18, bold=True)
    for row_index, model_key in enumerate(models):
        y = top + row_index * cell_h
        right_text(draw, (left - 22, y + cell_h / 2), model_labels[model_key], 21, bold=True)
        for column, task in enumerate(tasks):
            value = data[(model_key, task)]
            x = left + column * cell_w
            draw.rectangle(
                ((x + 2) * SCALE, (y + 2) * SCALE, (x + cell_w - 2) * SCALE, (y + cell_h - 2) * SCALE),
                fill=color(value),
                outline="white",
                width=2 * SCALE,
            )
            centered(draw, (x + cell_w / 2, y + cell_h / 2), f"{100 * value:+.1f}", 21, bold=True, fill="white" if abs(value) >= 0.55 else "#172033")
    base_y = top + len(models) * cell_h + 52
    centered(draw, (width / 2, base_y), "Red: grounding reduced success    |    Blue: grounding increased success", 20, bold=True)
    centered(draw, (width / 2, base_y + 40), "The 1B Granite row is an integration failure and is not interpreted as model capability.", 18, fill="#7d2934")
    save(image, "task_grounding_heatmap.png")


def main() -> None:
    render_readiness()
    render_headroom()
    render_heatmap()
    print("MANUSCRIPT_PNGS_READY", {"figures": 3, "scale": SCALE})


if __name__ == "__main__":
    main()
