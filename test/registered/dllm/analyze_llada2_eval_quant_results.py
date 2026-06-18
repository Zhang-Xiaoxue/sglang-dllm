#!/usr/bin/env python3
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

KEY_COLUMNS = ["model_size", "eval_name", "bs", "tp", "ep", "dp", "moe_a2a_backend"]
NUMERIC_COLUMNS = [
    "bs",
    "tp",
    "ep",
    "dp",
    "moe_dp_size",
    "max_running_requests",
    "num_examples",
    "num_threads",
    "max_tokens",
    "score",
    "mean_score",
    "latency",
    "output_throughput",
    "invalid",
    "chars",
]
SUMMARY_COLUMNS = [
    *KEY_COLUMNS,
    "bf16_score",
    "int8_score",
    "score_delta",
    "bf16_latency",
    "int8_latency",
    "latency_speedup",
    "latency_reduction_pct",
    "bf16_output_throughput",
    "int8_output_throughput",
    "throughput_speedup",
    "bf16_chars",
    "int8_chars",
    "chars_delta",
]

COLORS = {
    "bf16": "#2563eb",
    "int8": "#dc2626",
    "grid": "#d1d5db",
    "axis": "#374151",
    "text": "#111827",
    "muted": "#6b7280",
    "good": "#16a34a",
    "bad": "#dc2626",
    "neutral": "#f3f4f6",
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


def format_value(value, digits=4):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


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


def safe_div(a, b):
    if a is None or b in (None, 0):
        return None
    return a / b


def build_pairs(rows):
    by_key_precision = defaultdict(dict)
    for row in rows:
        if row.get("status") and row.get("status") != "passed":
            continue
        precision = row.get("precision")
        if precision in ("bf16", "int8"):
            by_key_precision[key_for(row)][precision] = row

    pairs = []
    missing = []
    for key, items in sorted(by_key_precision.items()):
        if "bf16" in items and "int8" in items:
            bf16 = items["bf16"]
            int8 = items["int8"]
            latency_speedup = safe_div(bf16.get("latency"), int8.get("latency"))
            throughput_speedup = safe_div(
                int8.get("output_throughput"), bf16.get("output_throughput")
            )
            pairs.append(
                {
                    **dict(zip(KEY_COLUMNS, key)),
                    "bf16_score": bf16.get("score"),
                    "int8_score": int8.get("score"),
                    "score_delta": (
                        int8.get("score") - bf16.get("score")
                        if int8.get("score") is not None and bf16.get("score") is not None
                        else None
                    ),
                    "bf16_latency": bf16.get("latency"),
                    "int8_latency": int8.get("latency"),
                    "latency_speedup": latency_speedup,
                    "latency_reduction_pct": (
                        (1 - int8.get("latency") / bf16.get("latency")) * 100
                        if int8.get("latency") is not None and bf16.get("latency")
                        else None
                    ),
                    "bf16_output_throughput": bf16.get("output_throughput"),
                    "int8_output_throughput": int8.get("output_throughput"),
                    "throughput_speedup": throughput_speedup,
                    "bf16_chars": bf16.get("chars"),
                    "int8_chars": int8.get("chars"),
                    "chars_delta": (
                        int8.get("chars") - bf16.get("chars")
                        if int8.get("chars") is not None and bf16.get("chars") is not None
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
        if row.get("status") and row.get("status") != "passed":
            continue
        if row.get(metric) is not None and row.get("precision") in ("bf16", "int8"):
            by_tp[int(row["tp"])].append(row)

    tps = sorted(by_tp)
    if not tps:
        return

    panel_w, panel_h = 560, 220
    margin_l, margin_r, margin_t, margin_b = 56, 22, 48, 42
    width = panel_w
    height = panel_h * len(tps) + margin_t + margin_b
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 24, title, 18, weight="bold"),
        svg_text(14, height / 2, ylabel, 12, anchor="middle", color=COLORS["muted"]),
    ]

    for panel_i, tp in enumerate(tps):
        rows_tp = by_tp[tp]
        top = margin_t + panel_i * panel_h
        left = margin_l
        right = width - margin_r
        bottom = top + panel_h - margin_b
        plot_top = top + 30

        bss = sorted({int(row["bs"]) for row in rows_tp})
        values = [row[metric] for row in rows_tp if row.get(metric) is not None]
        y_min, y_max = min(values), max(values)
        pad = (y_max - y_min) * 0.12 or max(abs(y_max) * 0.08, 1.0)
        y_min -= pad
        y_max += pad

        body.append(svg_text(left, top + 15, f"tp={tp}", 13, anchor="start", weight="bold"))
        body.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="{COLORS["axis"]}"/>')
        body.append(f'<line x1="{left}" y1="{plot_top}" x2="{left}" y2="{bottom}" stroke="{COLORS["axis"]}"/>')

        for tick in range(5):
            y = plot_top + (bottom - plot_top) * tick / 4
            value = y_max - (y_max - y_min) * tick / 4
            body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{COLORS["grid"]}" stroke-dasharray="3 3"/>')
            body.append(svg_text(left - 8, y + 4, format_value(value, 2), 10, anchor="end", color=COLORS["muted"]))

        for bs in bss:
            x = scale(bs, min(bss), max(bss), left, right)
            body.append(svg_text(x, bottom + 20, str(bs), 10, color=COLORS["muted"]))
        body.append(svg_text((left + right) / 2, bottom + 36, "bs", 11, color=COLORS["muted"]))

        for precision in ("bf16", "int8"):
            pts = []
            for row in sorted([r for r in rows_tp if r.get("precision") == precision], key=lambda r: int(r["bs"])):
                x = scale(int(row["bs"]), min(bss), max(bss), left, right)
                y = scale(row[metric], y_min, y_max, bottom, plot_top)
                pts.append((x, y, row[metric]))
            if len(pts) >= 2:
                body.append(
                    '<polyline fill="none" stroke="{}" stroke-width="2.2" points="{}"/>'.format(
                        COLORS[precision], " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in pts)
                    )
                )
            for x, y, value in pts:
                body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{COLORS[precision]}"/>')
                body.append(svg_text(x, y - 8, format_value(value, 3), 10, color=COLORS[precision]))

    legend_y = height - 10
    body.append(f'<circle cx="{width - 130}" cy="{legend_y - 4}" r="4" fill="{COLORS["bf16"]}"/>')
    body.append(svg_text(width - 118, legend_y, "bf16", 11, anchor="start"))
    body.append(f'<circle cx="{width - 70}" cy="{legend_y - 4}" r="4" fill="{COLORS["int8"]}"/>')
    body.append(svg_text(width - 58, legend_y, "int8", 11, anchor="start"))
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def heatmap(path, pairs, metric, title, center=None, suffix=""):
    rows = [row for row in pairs if row.get(metric) is not None]
    if not rows:
        return
    bss = sorted({int(row["bs"]) for row in rows})
    tps = sorted({int(row["tp"]) for row in rows})
    cell_w, cell_h = 90, 42
    margin_l, margin_t = 70, 58
    width = margin_l + len(tps) * cell_w + 36
    height = margin_t + len(bss) * cell_h + 48
    values = [row[metric] for row in rows]
    if center is None:
        v_min, v_max = min(values), max(values)
        center = 0 if v_min < 0 < v_max else None
    else:
        v_min, v_max = min(values + [center]), max(values + [center])

    def color(value):
        if center is None:
            ratio = 0.5 if v_max == v_min else (value - v_min) / (v_max - v_min)
            red = int(239 - ratio * 175)
            green = int(246 - ratio * 80)
            blue = int(255 - ratio * 150)
            return f"rgb({red},{green},{blue})"
        if value >= center:
            ratio = 0 if v_max == center else (value - center) / (v_max - center)
            red = int(240 - ratio * 180)
            green = int(253 - ratio * 100)
            blue = int(244 - ratio * 170)
            return f"rgb({red},{green},{blue})"
        ratio = 0 if v_min == center else (center - value) / (center - v_min)
        red = int(254 - ratio * 70)
        green = int(242 - ratio * 120)
        blue = int(242 - ratio * 120)
        return f"rgb({red},{green},{blue})"

    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 24, title, 18, weight="bold"),
        svg_text(margin_l - 42, margin_t - 15, "bs/tp", 11, color=COLORS["muted"]),
    ]
    for i, tp in enumerate(tps):
        x = margin_l + i * cell_w + cell_w / 2
        body.append(svg_text(x, margin_t - 16, f"tp={tp}", 11, weight="bold"))
    by_cell = {(int(row["bs"]), int(row["tp"])): row for row in rows}
    for j, bs in enumerate(bss):
        y = margin_t + j * cell_h
        body.append(svg_text(margin_l - 16, y + cell_h / 2 + 4, f"bs={bs}", 11, anchor="end", weight="bold"))
        for i, tp in enumerate(tps):
            x = margin_l + i * cell_w
            row = by_cell.get((bs, tp))
            if row is None:
                fill = COLORS["neutral"]
                text = "-"
            else:
                value = row[metric]
                fill = color(value)
                text = f"{value:.3f}{suffix}"
            body.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{fill}" stroke="white"/>')
            body.append(svg_text(x + cell_w / 2, y + cell_h / 2 + 4, text, 12, weight="bold"))
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def values(rows, metric):
    return [row[metric] for row in rows if row.get(metric) is not None]


def write_report(path, csv_path, pairs, missing, charts):
    throughput = values(pairs, "throughput_speedup")
    latency = values(pairs, "latency_speedup")
    score_delta = values(pairs, "score_delta")
    lines = [
        "# LLaDA2 Eval INT8 Quant Benefit Report",
        "",
        f"- Source CSV: `{csv_path}`",
        f"- Matched bf16/int8 cases: {len(pairs)}",
        "",
        "## Summary",
        "",
        f"- INT8 output throughput wins: {sum(1 for x in throughput if x > 1)}/{len(throughput)}",
        f"- Mean throughput speedup: {statistics.mean(throughput):.4f}x" if throughput else "- Mean throughput speedup: n/a",
        f"- Median throughput speedup: {statistics.median(throughput):.4f}x" if throughput else "- Median throughput speedup: n/a",
        f"- Mean latency speedup: {statistics.mean(latency):.4f}x" if latency else "- Mean latency speedup: n/a",
        f"- Median latency speedup: {statistics.median(latency):.4f}x" if latency else "- Median latency speedup: n/a",
        f"- Mean score delta: {statistics.mean(score_delta):+.4f}" if score_delta else "- Mean score delta: n/a",
        f"- Median score delta: {statistics.median(score_delta):+.4f}" if score_delta else "- Median score delta: n/a",
        "",
        "## Charts",
        "",
    ]
    for title, chart_path in charts.items():
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"![{title}]({chart_path.name})")
        lines.append("")
    lines += [
        "## Per-Case Table",
        "",
        "| eval | bs | tp | bf16 score | int8 score | score delta | bf16 tok/s | int8 tok/s | speedup | bf16 latency s | int8 latency s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(pairs, key=lambda r: (str(r["eval_name"]), int(r["tp"]), int(r["bs"]))):
        lines.append(
            f"| {row['eval_name']} | {row['bs']} | {row['tp']} | "
            f"{row['bf16_score']:.4f} | {row['int8_score']:.4f} | {row['score_delta']:+.4f} | "
            f"{row['bf16_output_throughput']:.2f} | {row['int8_output_throughput']:.2f} | "
            f"{row['throughput_speedup']:.4f}x | {row['bf16_latency']:.2f} | {row['int8_latency']:.2f} |"
        )
    if missing:
        lines += ["", "## Unmatched Cases", ""]
        for key, precisions in missing:
            lines.append(f"- {dict(zip(KEY_COLUMNS, key))}: {precisions}")
    lines += [
        "",
        "## Notes",
        "",
        "- `throughput_speedup = int8_output_throughput / bf16_output_throughput`.",
        "- `latency_speedup = bf16_latency / int8_latency`; larger than 1 means INT8 is faster.",
        "- GPQA/PIQA scores can move by one or more questions between runs; repeat key configs before treating small deltas as accuracy regressions.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    csv_path = Path(args.csv_path).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else csv_path.with_name("analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(csv_path)
    pairs, missing = build_pairs(rows)
    if not pairs:
        raise SystemExit("No matched bf16/int8 pairs found")

    summary_csv = out_dir / "quant_benefit_summary.csv"
    write_summary_csv(summary_csv, pairs)

    charts = {
        "Score by tp": out_dir / "score_by_tp.svg",
        "Output throughput by tp": out_dir / "output_throughput_by_tp.svg",
        "Latency by tp": out_dir / "latency_by_tp.svg",
        "INT8 / BF16 throughput speedup": out_dir / "throughput_speedup_heatmap.svg",
        "INT8 - BF16 score delta": out_dir / "score_delta_heatmap.svg",
    }
    line_chart_by_tp(charts["Score by tp"], rows, "score", "Eval score by tp", "score")
    line_chart_by_tp(
        charts["Output throughput by tp"],
        rows,
        "output_throughput",
        "Output throughput by tp",
        "tokens/s",
    )
    line_chart_by_tp(charts["Latency by tp"], rows, "latency", "Eval latency by tp", "seconds")
    heatmap(
        charts["INT8 / BF16 throughput speedup"],
        pairs,
        "throughput_speedup",
        "INT8 / BF16 throughput speedup",
        center=1,
        suffix="x",
    )
    heatmap(
        charts["INT8 - BF16 score delta"],
        pairs,
        "score_delta",
        "INT8 - BF16 score delta",
        center=0,
    )

    report = out_dir / "quant_benefit_report.md"
    write_report(report, csv_path, pairs, missing, charts)

    print(f"summary_csv={summary_csv}")
    print(f"report={report}")
    for chart_path in charts.values():
        print(f"chart={chart_path}")


if __name__ == "__main__":
    main()
