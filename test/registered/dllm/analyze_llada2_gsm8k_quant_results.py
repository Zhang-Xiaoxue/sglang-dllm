#!/usr/bin/env python3
import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


KEY_COLUMNS = ["model_size", "bs", "tp", "ep", "dp", "moe_a2a_backend"]
NUMERIC_COLUMNS = [
    "bs",
    "tp",
    "ep",
    "dp",
    "moe_dp_size",
    "max_running_requests",
    "bs_latency",
    "bs_tokens",
    "bs_acc_length",
    "bs_speed",
    "gsm8k_acc",
    "gsm8k_invalid",
    "gsm8k_latency",
    "gsm8k_output_throughput",
]
SUMMARY_COLUMNS = [
    *KEY_COLUMNS,
    "bf16_bs_speed",
    "int8_bs_speed",
    "bs_speedup",
    "bf16_gsm8k_throughput",
    "int8_gsm8k_throughput",
    "gsm8k_speedup",
    "bf16_bs_latency_per_token_ms",
    "int8_bs_latency_per_token_ms",
    "latency_per_token_reduction_pct",
    "bf16_acc",
    "int8_acc",
    "acc_delta",
    "bf16_invalid",
    "int8_invalid",
    "invalid_delta",
]


COLORS = {
    "bf16": "#2563eb",
    "int8": "#dc2626",
    "grid": "#d1d5db",
    "axis": "#374151",
    "text": "#111827",
    "muted": "#6b7280",
}


def to_number(value):
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return int(number)
    return number


def read_rows(path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for col in NUMERIC_COLUMNS:
            if col in row:
                row[col] = to_number(row[col])
    return rows


def key_for(row):
    return tuple(row.get(col, "") for col in KEY_COLUMNS)


def is_warmup(row):
    return str(row.get("run_name", "")).startswith("warmup")


def safe_div(a, b):
    if a is None or b in (None, 0):
        return None
    return a / b


def latency_per_token_ms(row):
    ratio = safe_div(row.get("bs_latency"), row.get("bs_tokens"))
    return ratio * 1000 if ratio is not None else None


def is_empty_result(row):
    return all(
        row.get(col) is None
        for col in ("bs_speed", "gsm8k_output_throughput", "gsm8k_acc")
    )


def build_pairs(rows, include_warmup=False):
    by_key_precision = defaultdict(dict)
    for row in rows:
        if not include_warmup and is_warmup(row):
            continue
        precision = row.get("precision")
        if precision not in ("bf16", "int8"):
            continue
        if is_empty_result(row):
            continue
        by_key_precision[key_for(row)][precision] = row

    pairs = []
    missing = []
    for key, items in sorted(by_key_precision.items()):
        if "bf16" in items and "int8" in items:
            bf16 = items["bf16"]
            int8 = items["int8"]
            bf16_lpt = latency_per_token_ms(bf16)
            int8_lpt = latency_per_token_ms(int8)
            pairs.append(
                {
                    **dict(zip(KEY_COLUMNS, key)),
                    "bf16_bs_speed": bf16.get("bs_speed"),
                    "int8_bs_speed": int8.get("bs_speed"),
                    "bs_speedup": safe_div(int8.get("bs_speed"), bf16.get("bs_speed")),
                    "bf16_gsm8k_throughput": bf16.get("gsm8k_output_throughput"),
                    "int8_gsm8k_throughput": int8.get("gsm8k_output_throughput"),
                    "gsm8k_speedup": safe_div(
                        int8.get("gsm8k_output_throughput"),
                        bf16.get("gsm8k_output_throughput"),
                    ),
                    "bf16_bs_latency_per_token_ms": bf16_lpt,
                    "int8_bs_latency_per_token_ms": int8_lpt,
                    "latency_per_token_reduction_pct": (
                        (1 - int8_lpt / bf16_lpt) * 100 if bf16_lpt else None
                    ),
                    "bf16_acc": bf16.get("gsm8k_acc"),
                    "int8_acc": int8.get("gsm8k_acc"),
                    "acc_delta": (
                        int8.get("gsm8k_acc") - bf16.get("gsm8k_acc")
                        if int8.get("gsm8k_acc") is not None
                        and bf16.get("gsm8k_acc") is not None
                        else None
                    ),
                    "bf16_invalid": bf16.get("gsm8k_invalid"),
                    "int8_invalid": int8.get("gsm8k_invalid"),
                    "invalid_delta": (
                        int8.get("gsm8k_invalid") - bf16.get("gsm8k_invalid")
                        if int8.get("gsm8k_invalid") is not None
                        and bf16.get("gsm8k_invalid") is not None
                        else None
                    ),
                }
            )
        else:
            missing.append((key, sorted(items)))
    return pairs, missing


def write_summary_csv(path, pairs):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in pairs:
            writer.writerow({col: format_value(row.get(col)) for col in SUMMARY_COLUMNS})


def format_value(value, digits=4):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def metric_values(rows, metric):
    return [row[metric] for row in rows if row.get(metric) is not None]


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def pct(value):
    return "" if value is None else f"{value * 100:.1f}%"


def escape_xml(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def svg_text(x, y, text, size=12, anchor="middle", color=None, weight="normal"):
    color = color or COLORS["text"]
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" fill="{color}" '
        f'font-weight="{weight}">{escape_xml(text)}</text>'
    )


def scale(value, old_min, old_max, new_min, new_max):
    if old_max == old_min:
        return (new_min + new_max) / 2
    return new_min + (value - old_min) * (new_max - new_min) / (old_max - old_min)


def line_chart_by_tp(path, rows, metric, title, ylabel):
    by_tp = defaultdict(list)
    for row in rows:
        if row.get(metric) is not None:
            by_tp[int(row["tp"])] .append(row)
    tps = sorted(by_tp)
    panel_w, panel_h = 640, 220
    margin_l, margin_r, margin_t, margin_b = 58, 24, 38, 40
    width = panel_w
    height = 42 + panel_h * len(tps)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 24, title, 16, weight="bold"),
    ]
    for panel_idx, tp in enumerate(tps):
        top = 42 + panel_idx * panel_h
        data = sorted(by_tp[tp], key=lambda r: int(r["bs"]))
        bs_values = sorted({int(r["bs"]) for r in data})
        series = {"bf16": {}, "int8": {}}
        for r in data:
            series[r["precision"]][int(r["bs"])] = r[metric]
        vals = [v for s in series.values() for v in s.values() if v is not None]
        ymin = min(vals) * 0.92 if vals else 0
        ymax = max(vals) * 1.08 if vals else 1
        x0, x1 = margin_l, width - margin_r
        y0, y1 = top + margin_t, top + panel_h - margin_b
        parts.append(svg_text(12, top + 20, f"tp={tp}", 12, anchor="start", weight="bold"))
        for i in range(5):
            y = y1 - i * (y1 - y0) / 4
            val = ymin + i * (ymax - ymin) / 4
            parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
            parts.append(svg_text(x0 - 8, y + 4, f"{val:.0f}", 10, anchor="end", color=COLORS["muted"]))
        parts.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{COLORS["axis"]}"/>')
        parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="{COLORS["axis"]}"/>')
        x_positions = {}
        for i, bs in enumerate(bs_values):
            x = scale(i, 0, max(len(bs_values) - 1, 1), x0, x1)
            x_positions[bs] = x
            parts.append(svg_text(x, y1 + 22, bs, 11, color=COLORS["muted"]))
        parts.append(svg_text((x0 + x1) / 2, top + panel_h - 6, "bs", 11, color=COLORS["muted"]))
        for precision in ("bf16", "int8"):
            points = []
            for bs in bs_values:
                if bs in series[precision]:
                    x = x_positions[bs]
                    y = scale(series[precision][bs], ymin, ymax, y1, y0)
                    points.append((x, y))
            if len(points) >= 2:
                point_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
                parts.append(f'<polyline points="{point_str}" fill="none" stroke="{COLORS[precision]}" stroke-width="2.2"/>')
            for x, y in points:
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{COLORS[precision]}"/>')
    legend_y = 22
    parts.append(f'<circle cx="{width-140}" cy="{legend_y-4}" r="4" fill="{COLORS["bf16"]}"/>')
    parts.append(svg_text(width - 130, legend_y, "bf16", 11, anchor="start"))
    parts.append(f'<circle cx="{width-80}" cy="{legend_y-4}" r="4" fill="{COLORS["int8"]}"/>')
    parts.append(svg_text(width - 70, legend_y, "int8", 11, anchor="start"))
    parts.append(svg_text(12, height / 2, ylabel, 11, anchor="middle", color=COLORS["muted"]))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def color_for_speedup(value):
    if value is None:
        return "#f3f4f6"
    if value >= 1:
        t = min((value - 1) / 0.8, 1)
        r = int(220 - 160 * t)
        g = int(252 - 80 * t)
        b = int(231 - 120 * t)
        return f"#{r:02x}{g:02x}{b:02x}"
    t = min((1 - value) / 0.4, 1)
    r = int(254 - 20 * t)
    g = int(226 - 140 * t)
    b = int(226 - 140 * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def heatmap(path, pairs, metric, title, suffix="x"):
    bs_values = sorted({int(r["bs"]) for r in pairs})
    tp_values = sorted({int(r["tp"]) for r in pairs})
    lookup = {(int(r["bs"]), int(r["tp"])): r.get(metric) for r in pairs}
    cell_w, cell_h = 88, 38
    left, top = 72, 52
    width = left + cell_w * len(tp_values) + 24
    height = top + cell_h * len(bs_values) + 44
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 24, title, 16, weight="bold"),
        svg_text(20, top - 12, "bs", 12, anchor="start", weight="bold"),
    ]
    for j, tp in enumerate(tp_values):
        x = left + j * cell_w + cell_w / 2
        parts.append(svg_text(x, top - 14, f"tp={tp}", 12, weight="bold"))
    for i, bs in enumerate(bs_values):
        y = top + i * cell_h
        parts.append(svg_text(left - 12, y + cell_h / 2 + 4, bs, 12, anchor="end"))
        for j, tp in enumerate(tp_values):
            x = left + j * cell_w
            value = lookup.get((bs, tp))
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{color_for_speedup(value)}" stroke="white"/>')
            label = "" if value is None else f"{value:.2f}{suffix}"
            parts.append(svg_text(x + cell_w / 2, y + cell_h / 2 + 4, label, 12))
    parts.append(svg_text(width / 2, height - 12, "green > 1 means int8 is faster; red < 1 means slower", 11, color=COLORS["muted"]))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def markdown_table(rows, columns, max_rows=None):
    if max_rows:
        rows = rows[:max_rows]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(format_value(row.get(col)) for col in columns) + " |")
    return "\n".join([header, sep, *body])


def group_average(pairs, group_col, metric):
    groups = defaultdict(list)
    for row in pairs:
        groups[row[group_col]].append(row.get(metric))
    return [
        {group_col: key, metric: mean(values)}
        for key, values in sorted(groups.items(), key=lambda kv: int(kv[0]))
    ]


def write_report(path, source_csv, pairs, missing, artifacts):
    bs_speedups = metric_values(pairs, "bs_speedup")
    gsm_speedups = metric_values(pairs, "gsm8k_speedup")
    acc_deltas = metric_values(pairs, "acc_delta")
    winners = [r for r in pairs if r.get("bs_speedup") is not None and r["bs_speedup"] > 1]
    best = max(pairs, key=lambda r: r.get("bs_speedup") or -1) if pairs else None
    worst = min(pairs, key=lambda r: r.get("bs_speedup") or 999) if pairs else None

    lines = [
        "# LLaDA2 BF16 vs INT8 Quantization Analysis",
        "",
        f"Source CSV: `{source_csv}`",
        f"Matched bf16/int8 cases: **{len(pairs)}**",
        f"INT8 wins on bs_speed: **{len(winners)}/{len(pairs)}**",
        "",
        "## Key numbers",
        "",
        f"- Median bs_speed speedup: **{format_value(median(bs_speedups))}x**",
        f"- Mean bs_speed speedup: **{format_value(mean(bs_speedups))}x**",
        f"- Median GSM8K throughput speedup: **{format_value(median(gsm_speedups))}x**",
        f"- Mean GSM8K throughput speedup: **{format_value(mean(gsm_speedups))}x**",
        f"- Median accuracy delta: **{format_value(median(acc_deltas))}**",
        "",
    ]
    if best:
        lines.append(
            "- Best bs_speed case: "
            f"bs={best['bs']}, tp={best['tp']}, speedup={format_value(best['bs_speedup'])}x "
            f"({format_value(best['bf16_bs_speed'])} -> {format_value(best['int8_bs_speed'])} tok/s)"
        )
    if worst:
        lines.append(
            "- Worst bs_speed case: "
            f"bs={worst['bs']}, tp={worst['tp']}, speedup={format_value(worst['bs_speedup'])}x "
            f"({format_value(worst['bf16_bs_speed'])} -> {format_value(worst['int8_bs_speed'])} tok/s)"
        )
    lines += [
        "",
        "## Visualizations",
        "",
        f"- [BS speed by tp]({artifacts['bs_speed_svg'].name})",
        f"- [GSM8K throughput by tp]({artifacts['gsm_svg'].name})",
        f"- [BS speedup heatmap]({artifacts['bs_heatmap'].name})",
        f"- [GSM8K speedup heatmap]({artifacts['gsm_heatmap'].name})",
        "",
        "## Speedup by tp",
        "",
        markdown_table(group_average(pairs, "tp", "bs_speedup"), ["tp", "bs_speedup"]),
        "",
        "## Speedup by bs",
        "",
        markdown_table(group_average(pairs, "bs", "bs_speedup"), ["bs", "bs_speedup"]),
        "",
        "## Matched case summary",
        "",
        markdown_table(
            sorted(pairs, key=lambda r: (int(r["tp"]), int(r["bs"]))),
            [
                "bs",
                "tp",
                "bf16_bs_speed",
                "int8_bs_speed",
                "bs_speedup",
                "bf16_gsm8k_throughput",
                "int8_gsm8k_throughput",
                "gsm8k_speedup",
                "acc_delta",
            ],
        ),
        "",
        "## How to read this",
        "",
        "- `bs_speedup = int8_bs_speed / bf16_bs_speed`. Values above 1 mean INT8 is faster.",
        "- `gsm8k_speedup = int8_gsm8k_output_throughput / bf16_gsm8k_output_throughput`.",
        "- Latency is normalized as latency per output token because output token counts differ across runs.",
        "- Warmup rows are excluded by default.",
    ]
    if missing:
        lines += ["", "## Unmatched cases", ""]
        for key, precisions in missing:
            lines.append(f"- {dict(zip(KEY_COLUMNS, key))}: {precisions}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=str(Path(__file__).with_name("llada2_mini_gsm8k_bf16_int8.csv")),
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--include-warmup", action="store_true")
    args = parser.parse_args()

    csv_path = Path(args.csv_path).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else csv_path.with_name("quant_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(csv_path)
    pairs, missing = build_pairs(rows, include_warmup=args.include_warmup)
    raw_rows = [row for row in rows if row.get("precision") in ("bf16", "int8") and (args.include_warmup or not is_warmup(row))]

    summary_csv = out_dir / "quant_benefit_summary.csv"
    write_summary_csv(summary_csv, pairs)
    bs_speed_svg = out_dir / "bs_speed_by_tp.svg"
    gsm_svg = out_dir / "gsm8k_throughput_by_tp.svg"
    bs_heatmap = out_dir / "bs_speedup_heatmap.svg"
    gsm_heatmap = out_dir / "gsm8k_speedup_heatmap.svg"
    line_chart_by_tp(bs_speed_svg, raw_rows, "bs_speed", "BS speed TPS by tp", "tokens/s")
    line_chart_by_tp(gsm_svg, raw_rows, "gsm8k_output_throughput", "GSM8K output throughput by tp", "tokens/s")
    heatmap(bs_heatmap, pairs, "bs_speedup", "INT8 / BF16 BS speedup")
    heatmap(gsm_heatmap, pairs, "gsm8k_speedup", "INT8 / BF16 GSM8K throughput speedup")

    report = out_dir / "quant_benefit_report.md"
    write_report(
        report,
        csv_path,
        pairs,
        missing,
        {
            "bs_speed_svg": bs_speed_svg,
            "gsm_svg": gsm_svg,
            "bs_heatmap": bs_heatmap,
            "gsm_heatmap": gsm_heatmap,
        },
    )
    print(f"Wrote {summary_csv}")
    print(f"Wrote {report}")
    print(f"Wrote {bs_speed_svg}")
    print(f"Wrote {gsm_svg}")
    print(f"Wrote {bs_heatmap}")
    print(f"Wrote {gsm_heatmap}")


if __name__ == "__main__":
    main()
