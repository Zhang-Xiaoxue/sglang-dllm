#!/usr/bin/env python3
"""Build a portable Data Analytics report from an EP backend=none CSV."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


INT_FIELDS = ("bs", "tp", "ep", "dp", "moe_dp_size", "max_running_requests")
FLOAT_FIELDS = (
    "bs_latency",
    "bs_tokens",
    "bs_acc_length",
    "bs_speed",
    "gsm8k_acc",
    "gsm8k_invalid",
    "gsm8k_latency",
    "gsm8k_output_throughput",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        for field in INT_FIELDS:
            row[field] = int(row[field]) if row.get(field) else None
        for field in FLOAT_FIELDS:
            row[field] = float(row[field]) if row.get(field) else None
    return rows


def case_key(row: dict) -> tuple:
    return (
        row["bs"],
        row["tp"],
        row["ep"],
        row["dp"],
        row["moe_a2a_backend"],
    )


def rounded(value: float | None, digits: int = 4) -> float | None:
    return round(value, digits) if value is not None else None


def mean(values: list[float]) -> float:
    return statistics.mean(values)


def build_artifact(rows: list[dict], csv_path: Path) -> dict:
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    complete = [
        row
        for row in rows
        if row["bs_speed"] is not None
        and row["gsm8k_output_throughput"] is not None
    ]
    incomplete = [row for row in rows if row not in complete]

    key_counts = Counter(case_key(row) for row in rows)
    duplicate_count = sum(count - 1 for count in key_counts.values() if count > 1)
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate configuration row(s)")

    expected = {
        (bs, tp, ep, 1, "none")
        for bs in (1, 4, 8, 16, 32, 64, 128, 256)
        for tp in (8, 4, 2, 1)
        for ep in (8, 4, 2, 1)
        if ep <= tp
    }
    missing_expected = expected - set(key_counts)
    unexpected = set(key_counts) - expected

    accuracies = [row["gsm8k_acc"] for row in complete]
    accuracy_mean = mean(accuracies)
    accuracy_min = min(accuracies)
    accuracy_max = max(accuracies)
    invalid_rows = [row for row in complete if row["gsm8k_invalid"]]
    binomial_margin = 1.96 * math.sqrt(accuracy_mean * (1 - accuracy_mean) / 200)

    accuracy_by_ep = {}
    for ep in (1, 2, 4, 8):
        values = [row["gsm8k_acc"] for row in complete if row["ep"] == ep]
        accuracy_by_ep[ep] = mean(values)

    best_by_bs = []
    best_lookup = {}
    for bs in sorted({row["bs"] for row in complete}):
        subset = [row for row in complete if row["bs"] == bs]
        best_fixed = max(subset, key=lambda row: row["bs_speed"])
        best_gsm = max(subset, key=lambda row: row["gsm8k_output_throughput"])
        best_lookup[bs] = {"fixed": best_fixed, "gsm": best_gsm}
        best_by_bs.append(
            {
                "bs": bs,
                "fixed_speed": rounded(best_fixed["bs_speed"], 2),
                "fixed_tp": best_fixed["tp"],
                "fixed_ep": best_fixed["ep"],
                "gsm_throughput": rounded(
                    best_gsm["gsm8k_output_throughput"], 2
                ),
                "gsm_latency": rounded(best_gsm["gsm8k_latency"], 2),
                "gsm_tp": best_gsm["tp"],
                "gsm_ep": best_gsm["ep"],
                "gsm_accuracy": rounded(best_gsm["gsm8k_acc"], 4),
            }
        )

    high_bs_ep = []
    for ep in (1, 2, 4, 8):
        item = {"ep": ep}
        for bs in (64, 128, 256):
            row = next(
                row
                for row in complete
                if row["bs"] == bs and row["tp"] == 8 and row["ep"] == ep
            )
            item[f"bs{bs}"] = rounded(row["gsm8k_output_throughput"], 2)
            item[f"fixed_bs{bs}"] = rounded(row["bs_speed"], 2)
            item[f"acc_bs{bs}"] = rounded(row["gsm8k_acc"], 4)
        high_bs_ep.append(item)

    fixed_peak = max(complete, key=lambda row: row["bs_speed"])
    gsm_peak = max(complete, key=lambda row: row["gsm8k_output_throughput"])
    tp4_bs128 = max(
        (
            row
            for row in complete
            if row["bs"] == 128 and row["tp"] == 4
        ),
        key=lambda row: row["gsm8k_output_throughput"],
    )
    tp8_bs128 = best_lookup[128]["gsm"]

    recommended_rows = []
    recommendations = (
        ("固定长度批量峰值", fixed_peak),
        ("GSM8K 绝对吞吐峰值", gsm_peak),
        ("4-NPU 资源效率", tp4_bs128),
    )
    for rank, (use_case, row) in enumerate(recommendations, 1):
        recommended_rows.append(
            {
                "rank": rank,
                "use_case": use_case,
                "bs": row["bs"],
                "tp": row["tp"],
                "ep": row["ep"],
                "bs_speed": rounded(row["bs_speed"], 2),
                "gsm_throughput": rounded(
                    row["gsm8k_output_throughput"], 2
                ),
                "gsm_latency": rounded(row["gsm8k_latency"], 2),
                "gsm_accuracy": rounded(row["gsm8k_acc"], 4),
                "gsm_throughput_per_npu": rounded(
                    row["gsm8k_output_throughput"] / row["tp"], 2
                ),
            }
        )

    failed_rows = [
        {
            "sort_key": row["bs"] * 100 + row["tp"] * 10 + row["ep"],
            "bs": row["bs"],
            "tp": row["tp"],
            "ep": row["ep"],
            "status": "empty metrics",
            "interpretation": "CSV 不含失败原因，需查对应运行日志",
        }
        for row in incomplete
    ]

    bs128_to_256_fixed = (
        best_lookup[256]["fixed"]["bs_speed"]
        / best_lookup[128]["fixed"]["bs_speed"]
        - 1
    )
    bs128_to_256_gsm = (
        best_lookup[256]["gsm"]["gsm8k_output_throughput"]
        / best_lookup[128]["gsm"]["gsm8k_output_throughput"]
        - 1
    )
    tp4_efficiency_gain = (
        tp4_bs128["gsm8k_output_throughput"] / tp4_bs128["tp"]
    ) / (tp8_bs128["gsm8k_output_throughput"] / tp8_bs128["tp"]) - 1

    tp8_bs128_ep1 = next(
        row
        for row in complete
        if row["bs"] == 128 and row["tp"] == 8 and row["ep"] == 1
    )
    tp8_bs128_ep8 = next(
        row
        for row in complete
        if row["bs"] == 128 and row["tp"] == 8 and row["ep"] == 8
    )
    ep8_fixed_drop = 1 - tp8_bs128_ep8["bs_speed"] / tp8_bs128_ep1["bs_speed"]
    ep8_gsm_drop = (
        1
        - tp8_bs128_ep8["gsm8k_output_throughput"]
        / tp8_bs128_ep1["gsm8k_output_throughput"]
    )

    csv_source = {
        "id": "benchmark_csv",
        "label": "LLaDA2.1-mini BF16 EP backend=none benchmark CSV",
        "path": str(csv_path),
        "query": {
            "engine": "local_csv",
            "language": "csv",
            "description": "80-case TP/EP/BS matrix benchmark on LLaDA2.1-mini with moe_a2a_backend=none.",
            "executed_at": generated_at,
            "filters": [
                "model_size=mini",
                "precision=bf16_ep",
                "dp=1",
                "moe_dp_size=1",
                "moe_a2a_backend=none",
                "complete metric rows only in performance charts",
            ],
            "metric_definitions": [
                "bs_speed = actual completion tokens divided by the maximum end-to-end latency in one batched /generate request.",
                "gsm8k_output_throughput = total generated GSM8K output tokens divided by evaluation latency.",
                "gsm8k_acc = exact-match accuracy over 200 GSM8K questions.",
                "gsm8k_invalid = fraction of GSM8K responses that could not be parsed as valid answers.",
            ],
            "tables_used": [csv_path.name],
        },
    }
    runtime_source = {
        "id": "runtime_code",
        "label": "LLaDA2 EP benchmark definitions",
        "query": {
            "engine": "local_source",
            "language": "python/bash",
            "description": "Test and matrix definitions used to interpret concurrency, token counts, and metrics.",
            "executed_at": generated_at,
            "tables_used": [
                "test/registered/dllm/test_llada2_ascend_gsm8k_ep_bf16.py",
                "test/registered/dllm/run_llada2_ascend_gsm8k_EP_test_csv.sh",
                "python/sglang/test/send_one.py",
            ],
        },
    }
    sources = [csv_source, runtime_source]

    charts = [
        {
            "id": "best_gsm_by_bs",
            "title": "各 BS 的最佳 GSM8K 吞吐",
            "subtitle": "每个 BS 在全部 TP/EP 配置中取最高值；客户端并发固定为 128",
            "intent": "comparison",
            "question": "提高服务端 BS 后，最佳 GSM8K 吞吐如何变化？",
            "rationale": "8 个离散 BS 档位适合用柱状图比较绝对吞吐和平台期。",
            "comparisonContext": {
                "baseline": "每个 BS 内取 GSM8K output throughput 最大的 TP/EP 配置",
                "denominator": "GSM8K 评测总耗时",
                "grain": "BS 最佳配置",
                "unit": "output token/s",
            },
            "type": "bar",
            "dataset": "best_by_bs",
            "sourceId": "benchmark_csv",
            "xField": "bs",
            "xAxisTitle": "Batch size / max running requests",
            "yAxisTitle": "Output token/s",
            "series": [
                {
                    "field": "gsm_throughput",
                    "label": "Best GSM8K throughput",
                    "color": "blue",
                    "role": "actual",
                }
            ],
            "valueFormat": "number",
            "unit": "token/s",
            "layout": "full",
            "palette": {"kind": "categorical"},
            "settings": {
                "groupMode": "grouped",
                "orientation": "vertical",
                "sort": "none",
                "showValues": False,
            },
            "labels": {"values": "auto"},
            "surface": {"surface": "export", "viewMode": "both"},
        },
        {
            "id": "tp8_high_bs_ep",
            "title": "高并发 TP=8 下的 EP 敏感性",
            "subtitle": "GSM8K output token/s；EP=8 在 BS=64/128/256 均未形成最佳点",
            "intent": "comparison",
            "question": "高并发时，增大 EP 是否持续提升 GSM8K 吞吐？",
            "rationale": "EP 是离散档位，分组柱状图可直接比较三个高并发 BS。",
            "comparisonContext": {
                "baseline": "同一 BS、TP=8 下比较 EP=1/2/4/8",
                "denominator": "GSM8K 评测总耗时",
                "grain": "BS × EP 单次测试",
                "unit": "output token/s",
            },
            "type": "bar",
            "dataset": "high_bs_ep",
            "sourceId": "benchmark_csv",
            "xField": "ep",
            "xAxisTitle": "Expert parallel size",
            "yAxisTitle": "Output token/s",
            "series": [
                {"field": "bs64", "label": "BS=64", "color": "blue", "role": "actual"},
                {"field": "bs128", "label": "BS=128", "color": "orange", "role": "comparison"},
                {"field": "bs256", "label": "BS=256", "color": "pink", "role": "comparison"},
            ],
            "valueFormat": "number",
            "unit": "token/s",
            "layout": "full",
            "palette": {"kind": "categorical"},
            "settings": {
                "groupMode": "grouped",
                "orientation": "vertical",
                "sort": "none",
                "showValues": False,
            },
            "labels": {"values": "auto"},
            "legend": {"position": "bottom", "sort": "spec", "title": "Batch size"},
            "surface": {"surface": "export", "viewMode": "both"},
        },
    ]

    tables = [
        {
            "id": "recommended_configs",
            "title": "建议优先复测的配置",
            "subtitle": "绝对吞吐、固定批量峰值与每 NPU 效率对应不同目标",
            "dataset": "recommended_configs",
            "defaultSort": {"field": "rank", "direction": "asc"},
            "density": "spacious",
            "sourceId": "benchmark_csv",
            "layout": "full",
            "columns": [
                {"field": "rank", "label": "#", "format": "number"},
                {"field": "use_case", "label": "目标", "type": "text"},
                {"field": "bs", "label": "BS", "format": "number"},
                {"field": "tp", "label": "TP", "format": "number"},
                {"field": "ep", "label": "EP", "format": "number"},
                {"field": "bs_speed", "label": "Fixed batch token/s", "format": "number"},
                {"field": "gsm_throughput", "label": "GSM8K token/s", "format": "number"},
                {"field": "gsm_latency", "label": "GSM8K latency (s)", "format": "number"},
                {"field": "gsm_accuracy", "label": "GSM8K accuracy", "format": "percent"},
                {"field": "gsm_throughput_per_npu", "label": "GSM token/s/NPU", "format": "number"},
            ],
        },
        {
            "id": "incomplete_configs",
            "title": "缺失指标的配置",
            "subtitle": "4 个 case 仅写入配置字段，CSV 无法区分启动失败、OOM 或测试异常",
            "dataset": "incomplete_configs",
            "defaultSort": {"field": "bs", "direction": "asc"},
            "density": "spacious",
            "sourceId": "benchmark_csv",
            "layout": "full",
            "columns": [
                {"field": "bs", "label": "BS", "format": "number"},
                {"field": "tp", "label": "TP", "format": "number"},
                {"field": "ep", "label": "EP", "format": "number"},
                {"field": "status", "label": "Status", "type": "text"},
                {"field": "interpretation", "label": "可得结论", "type": "text"},
            ],
        },
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": "# LLaDA2 Mini none+EP 基准分析"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": "benchmark_csv",
            "body": (
                "## 技术结论\n\n"
                f"`none+EP` 的正确性结果总体稳定：80 个计划配置中 76 个有完整指标，完整结果的 GSM8K 平均准确率为 **{accuracy_mean:.2%}**，"
                f"EP=1/2/4/8 的分组均值仅落在 **{min(accuracy_by_ep.values()):.2%}–{max(accuracy_by_ep.values()):.2%}**，没有随 EP 增大的退化趋势。"
                f"性能上，固定长度批量峰值是 **BS={fixed_peak['bs']}、TP={fixed_peak['tp']}、EP={fixed_peak['ep']}：{fixed_peak['bs_speed']:.1f} token/s**；"
                f"GSM8K 峰值是 **BS={gsm_peak['bs']}、TP={gsm_peak['tp']}、EP={gsm_peak['ep']}：{gsm_peak['gsm8k_output_throughput']:.1f} token/s**。"
                "结论不是“EP 越大越快”：高并发下 EP=1 或 EP=2 更稳，EP=8 通常已经被路由分散和通信开销反噬。"
            ),
        },
        {
            "id": "correctness",
            "type": "markdown",
            "sourceId": "benchmark_csv",
            "body": (
                "## 精度没有显示 EP 相关回归\n\n"
                f"完整 case 的准确率范围为 **{accuracy_min:.1%}–{accuracy_max:.1%}**，全部高于测试阈值 68%。"
                f"只有 **{len(invalid_rows)}** 个 case 出现 `gsm8k_invalid=0.005`，即各 200 道题中约 1 道无法解析；其余均为 0。"
                f"按 200 题二项分布粗略估算，单次准确率在当前均值附近的 95% 抽样波动约为 **±{binomial_margin:.1%}**，"
                "因此表内几个百分点的差异不足以证明 EP 改变模型精度。当前证据支持 `none+EP` 功能正确，但关键配置仍应重复运行并逐题比对输出。"
            ),
        },
        {
            "id": "scaling",
            "type": "markdown",
            "sourceId": "benchmark_csv",
            "body": (
                "## BS=128 后吞吐进入平台区\n\n"
                f"各 BS 取最佳配置后，从 BS=128 提高到 BS=256，固定长度批量吞吐 **{bs128_to_256_fixed:+.1%}**，"
                f"GSM8K 吞吐仅 **{bs128_to_256_gsm:+.1%}**。GSM8K 客户端并发固定为 128，"
                "所以 `max_running_requests=256` 并不代表实际存在 256 个并发请求；BS=256 更适合视为同一 128 并发负载下的调度上限实验。"
                "如果目标是吞吐与延迟的平衡，BS=128 已接近最佳区域。"
            ),
        },
        {"id": "best_gsm_chart", "type": "chart", "chartId": "best_gsm_by_bs", "layout": "full"},
        {
            "id": "ep_sensitivity",
            "type": "markdown",
            "sourceId": "benchmark_csv",
            "body": (
                "## 高并发下 EP=8 明显过度切分\n\n"
                f"在 BS=128、TP=8 时，EP=8 相比 EP=1 的固定批量吞吐下降 **{ep8_fixed_drop:.1%}**，GSM8K 吞吐下降 **{ep8_gsm_drop:.1%}**。"
                "BS=64 和 BS=256 也没有出现 EP=8 最优。EP 增大虽然减少每个 rank 的本地 expert 数，"
                "但同时缩小每个 expert 收到的 token 批量，并增加路由、同步和负载不均衡的相对成本；backend=none 下这种折中尤其依赖 BS。"
                "这是描述性解释，仍需逐层 profiler 验证通信与 expert GEMM 的实际占比。"
            ),
        },
        {"id": "ep_chart", "type": "chart", "chartId": "tp8_high_bs_ep", "layout": "full"},
        {
            "id": "resource_efficiency",
            "type": "markdown",
            "sourceId": "benchmark_csv",
            "body": (
                "## TP=8 追求绝对吞吐，TP=4 更省卡\n\n"
                f"BS=128 时，TP=8 的最佳 GSM8K 吞吐为 {tp8_bs128['gsm8k_output_throughput']:.1f} token/s，"
                f"TP=4 的最佳值为 {tp4_bs128['gsm8k_output_throughput']:.1f} token/s。TP=8 的绝对吞吐高 **{tp8_bs128['gsm8k_output_throughput']/tp4_bs128['gsm8k_output_throughput']-1:.1%}**，"
                f"但 TP=4 的每 NPU 吞吐高 **{tp4_efficiency_gain:.1%}**。因此部署决策要明确是追求单实例峰值，还是追求集群 token/s/NPU。"
            ),
        },
        {"id": "recommendation_table", "type": "table", "tableId": "recommended_configs", "layout": "full"},
        {
            "id": "data_quality",
            "type": "markdown",
            "sourceId": "benchmark_csv",
            "body": (
                "## 四个 TP=2 高并发 case 需要查日志\n\n"
                f"配置矩阵本身完整：共 {len(rows)} 行、无重复、无缺失配置键；但 {len(incomplete)} 行的全部性能和精度指标为空，"
                "集中在 BS=128/256、TP=2、EP=1/2。这个集中模式不是普通随机缺失，但 CSV 没有保存退出码或错误阶段，"
                "不能仅凭空值断言 OOM。另有 12 个完整 case 的 `bs_tokens` 低于 `BS×512`，最差为 96.2%；"
                "由于 `bs_speed` 使用实际生成 token 作为分子，这更像提前停止造成的工作量差异，需要在严格固定长度基准中显式禁用 EOS 或单独标注。"
            ),
        },
        {"id": "failed_table", "type": "table", "tableId": "incomplete_configs", "layout": "full"},
        {
            "id": "scope",
            "type": "markdown",
            "sourceId": "runtime_code",
            "body": (
                "## 范围与指标定义\n\n"
                "范围为 LLaDA2.1-mini、BF16、DP=1、MoE DP=1、`moe_a2a_backend=none`，BS=1–256、TP=1/2/4/8、EP≤TP。"
                "固定批量测试每条请求最多生成 512 token，`bs_speed` 是实际 completion token 总数除以批内最慢响应延迟。"
                "GSM8K 使用 200 道题、客户端并发 128、每题最多生成 512 token；服务端 `max_running_requests=BS`。"
                "两种吞吐测量不同请求长度和调度形态，不能把其中一个场景的最佳配置直接当作另一个场景的最佳值。"
            ),
        },
        {
            "id": "method",
            "type": "markdown",
            "sourceId": "benchmark_csv",
            "body": (
                "## 方法\n\n"
                "分析先验证 80 个期望配置键、重复行、空指标和字段取值，再在每个 BS 内分别选取固定批量与 GSM8K 的最佳配置。"
                "EP 效应只在相同 BS/TP 内比较；资源效率以 GSM8K output token/s 除以 TP 使用的 NPU 数计算。"
                "所有结论均为单次观测的描述性比较，没有把相关性写成因果关系。"
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## 局限与稳健性\n\n"
                "每个配置只有一次运行，没有均值、方差或置信区间；1%–5% 的性能差距可能属于运行抖动。"
                "CSV 未记录 graph 开关、`mem-fraction-static`、CANN/torch_npu/SGLang 提交版本、NPU 频率与温度，也没有失败日志路径。"
                "GSM8K 的并发固定为 128，使 BS=256 的解释受限。准确率的二项区间只是粗略参考，因为所有配置使用同一组题目，观测并非独立样本。"
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## 建议下一步\n\n"
                "- 最大吞吐优先复测 `BS=256, TP=8, EP=2`；固定批量峰值优先复测 `BS=128, TP=8, EP=1`。\n"
                "- 两个候选配置各重复至少 3 次，并保存均值、标准差、P50/P95 latency。\n"
                "- 资源效率场景复测 `BS=128, TP=4, EP=2`，与 TP=8 方案比较 token/s/NPU 和总功耗。\n"
                "- 对四个 TP=2 空指标 case 保存独立日志、退出码、峰值显存和失败阶段。\n"
                "- 下一轮 CSV 增加 graph 模式、mem fraction、软件版本、warmup 次数和日志路径字段，方便与 DeepEP 做严格配对。"
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## 仍待确认\n\n"
                "- TP=2 在 BS≥128 的失败是显存、graph bucket、通信还是调度上限问题？\n"
                "- BS=128 与 BS=256 的 1.4% GSM8K 差距能否在重复测试中保持？\n"
                "- DeepEP 在相同 BS/TP/EP、相同 graph 设置下能否超过 none，且保持逐题输出一致？"
            ),
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "LLaDA2 Mini none+EP 基准分析",
            "description": "LLaDA2.1-mini BF16、backend=none 的 80-case BS/TP/EP 性能与精度诊断",
            "generatedAt": generated_at,
            "sources": sources,
            "charts": charts,
            "tables": tables,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "best_by_bs": best_by_bs,
                "high_bs_ep": high_bs_ep,
                "recommended_configs": recommended_rows,
                "incomplete_configs": failed_rows,
            },
        },
        "sources": copy.deepcopy(sources),
    }

    artifact["analysis_notes"] = {
        "expected_cases": len(expected),
        "recorded_cases": len(rows),
        "complete_cases": len(complete),
        "incomplete_cases": len(incomplete),
        "missing_expected_keys": sorted(missing_expected),
        "unexpected_keys": sorted(unexpected),
        "accuracy_by_ep": {
            str(ep): rounded(value, 6) for ep, value in accuracy_by_ep.items()
        },
        "chart_map": [
            {
                "section": "BS=128 后吞吐进入平台区",
                "chart": "best_gsm_by_bs",
                "family": "comparison",
                "claim": "GSM8K throughput plateaus between BS=128 and BS=256.",
            },
            {
                "section": "高并发下 EP=8 明显过度切分",
                "chart": "tp8_high_bs_ep",
                "family": "comparison",
                "claim": "EP=8 is not the best TP=8 configuration at BS=64/128/256.",
            },
        ],
        "omissions": [
            "No causal claim: no profiler traces or repeated runs are available.",
            "No time-series chart: BS is a configuration axis, not time.",
            "Failure root causes are not classified because the CSV has no exit codes or log paths.",
        ],
    }
    return artifact


def main() -> None:
    args = parse_args()
    rows = load_rows(args.csv_path)
    artifact = build_artifact(rows, args.csv_path)
    analysis_notes = artifact.pop("analysis_notes")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output_dir / "artifact.json"
    notes_path = args.output_dir / "source_notes.json"
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    notes_path.write_text(
        json.dumps(analysis_notes, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(artifact_path)


if __name__ == "__main__":
    main()
