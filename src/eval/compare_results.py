import argparse
import glob
import json
import math
import os
from datetime import datetime


def safe_float(value):
    value = float(value)
    if math.isinf(value) or math.isnan(value):
        return 0.0
    return value


def load_rows(path):
    rows = []
    for filename in glob.glob(os.path.join(path, "log", "stats_*.json")):
        with open(filename, "r") as f:
            row = json.load(f)
        rows.append(row)
    rows.sort(key=lambda item: int(item["id"]))
    return rows


def summarize(rows):
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "success": 0,
            "oracle_success": 0,
            "spl_sum": 0.0,
            "success_rate": 0.0,
            "oracle_success_rate": 0.0,
            "spl": 0.0,
            "distance_to_goal": 0.0,
            "path_length": 0.0,
            "ndtw": 0.0,
            "oracle_only": 0,
        }
    success = sum(int(row["success"]) for row in rows)
    oracle_success = sum(int(row["oracle_success"]) for row in rows)
    spl_sum = sum(safe_float(row["spl"]) for row in rows)
    distance_to_goal = sum(safe_float(row["distance_to_goal"]) for row in rows) / n
    path_length = sum(safe_float(row["path_length"]) for row in rows) / n
    ndtw = sum(safe_float(row.get("ndtw", 0.0)) for row in rows) / n
    oracle_only = sum(int(row["oracle_success"]) and not int(row["success"]) for row in rows)
    return {
        "n": n,
        "success": success,
        "oracle_success": oracle_success,
        "spl_sum": spl_sum,
        "success_rate": success / n,
        "oracle_success_rate": oracle_success / n,
        "spl": spl_sum / n,
        "distance_to_goal": distance_to_goal,
        "path_length": path_length,
        "ndtw": ndtw,
        "oracle_only": oracle_only,
    }


def aligned_rows(candidate_rows, baseline_rows):
    baseline_by_id = {str(row["id"]): row for row in baseline_rows}
    candidate_aligned = []
    baseline_aligned = []
    for row in candidate_rows:
        key = str(row["id"])
        if key in baseline_by_id:
            candidate_aligned.append(row)
            baseline_aligned.append(baseline_by_id[key])
    return candidate_aligned, baseline_aligned


def decide(candidate, baseline):
    delta_sr = candidate["success_rate"] - baseline["success_rate"]
    delta_osr = candidate["oracle_success_rate"] - baseline["oracle_success_rate"]
    delta_spl = candidate["spl"] - baseline["spl"]
    delta_path = candidate["path_length"] - baseline["path_length"]
    delta_oracle_only = candidate["oracle_only"] - baseline["oracle_only"]

    if candidate["n"] == 0:
        return "invalid", "没有完成任何episode，无法判断。"
    if delta_sr > 0 and delta_spl >= -0.01 and delta_path <= 0.5 and delta_osr >= -0.01:
        return "promote", "SR高于同ID baseline，且SPL/Path/OSR没有明显退化，建议进入更大pilot或全量。"
    if delta_sr < 0 and delta_osr >= -0.01 and delta_oracle_only > 0:
        return "stop", "OSR基本不变但SR下降，oracle-only增加，说明stop/final action问题变严重。"
    if delta_osr < -0.01:
        return "stop", "OSR明显下降，说明路线选择被破坏。"
    if delta_path > 0.5:
        return "stop", "Path length明显变长，动作策略更拖沓。"
    if delta_spl < -0.01:
        return "stop", "SPL明显下降，效率退化。"
    return "continue", "指标接近baseline但没有明确提升，建议继续测试其他候选。"


def format_metric_block(title, summary):
    return "\n".join([
        f"{title}",
        f"Success rate: {summary['success']}/{summary['n']} ({summary['success_rate']:.3f})",
        f"Oracle success rate: {summary['oracle_success']}/{summary['n']} ({summary['oracle_success_rate']:.3f})",
        f"SPL: {summary['spl_sum']:.3f}/{summary['n']} ({summary['spl']:.3f})",
        f"Distance to goal: {summary['distance_to_goal']:.3f}",
        f"Path length: {summary['path_length']:.3f}",
        f"ndtw: {summary['ndtw']:.3f}",
        f"oracle_success=1, success=0: {summary['oracle_only']}/{summary['n']}",
    ])


def append_result(args, candidate, baseline, flips, decision, reason, delta):
    block = f"""


====================================
{datetime.now().strftime('%m%d %H:%M')} SR提升循环实验：{args.label}（{args.phase}）
{args.path}

评测环境 Python: {args.eval_python}
vLLM 服务 Python: {args.vllm_python}
vLLM 服务: GPU 3, port 8201, gpu_memory_utilization=0.9
Worker: {args.worker_count}
Worker GPUs: {args.worker_gpus}
Prompt style: {args.prompt_style}
Temperature: {args.temperature}
Max episodes per split: {args.max_episodes}

{format_metric_block('当前实验结果：', candidate)}

同ID baseline对比：
{args.baseline_path}
{format_metric_block('Baseline同ID结果：', baseline)}

差值（当前 - baseline）：
SR: {delta['success_rate']:+.3f}
OSR: {delta['oracle_success_rate']:+.3f}
SPL: {delta['spl']:+.3f}
Distance to goal: {delta['distance_to_goal']:+.3f}
Path length: {delta['path_length']:+.3f}
oracle-only: {delta['oracle_only']:+d}

Success翻转：
new success / baseline fail: {flips['new_win']}
new fail / baseline success: {flips['new_lose']}

分析结论：{reason}
下一步决策：{decision}
=====================================
"""
    with open(args.result_file, "a", encoding="utf-8") as f:
        f.write(block)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--baseline-path", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--result-file", default="result.txt")
    parser.add_argument("--eval-python", required=True)
    parser.add_argument("--vllm-python", required=True)
    parser.add_argument("--worker-count", required=True)
    parser.add_argument("--worker-gpus", required=True)
    parser.add_argument("--prompt-style", required=True)
    parser.add_argument("--temperature", required=True)
    parser.add_argument("--max-episodes", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    candidate_rows = load_rows(args.path)
    baseline_rows = load_rows(args.baseline_path)
    candidate_aligned, baseline_aligned = aligned_rows(candidate_rows, baseline_rows)
    candidate_summary = summarize(candidate_aligned)
    baseline_summary = summarize(baseline_aligned)

    baseline_by_id = {str(row["id"]): row for row in baseline_aligned}
    flips = {
        "new_win": sum(int(row["success"]) and not int(baseline_by_id[str(row["id"])]["success"]) for row in candidate_aligned),
        "new_lose": sum((not int(row["success"])) and int(baseline_by_id[str(row["id"])]["success"]) for row in candidate_aligned),
    }

    delta = {
        key: candidate_summary[key] - baseline_summary[key]
        for key in [
            "success_rate",
            "oracle_success_rate",
            "spl",
            "distance_to_goal",
            "path_length",
            "oracle_only",
        ]
    }
    decision, reason = decide(candidate_summary, baseline_summary)

    summary = {
        "label": args.label,
        "phase": args.phase,
        "path": args.path,
        "baseline_path": args.baseline_path,
        "candidate": candidate_summary,
        "baseline": baseline_summary,
        "delta": delta,
        "flips": flips,
        "decision": decision,
        "reason": reason,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.summary_json:
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    if args.append:
        append_result(args, candidate_summary, baseline_summary, flips, decision, reason, delta)


if __name__ == "__main__":
    main()
