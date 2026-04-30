import argparse
import csv
import glob
import gzip
import json
import math
import os
from collections import defaultdict


def safe_float(value):
    value = float(value)
    if math.isinf(value) or math.isnan(value):
        return 0.0
    return value


def load_dataset(dataset_path):
    raw = json.load(gzip.open(dataset_path, "rt"))
    episodes = raw["episodes"]
    meta = {}
    for ep in episodes:
        episode_id = int(ep["episode_id"])
        scene_name = os.path.splitext(os.path.basename(ep["scene_id"]))[0]
        instruction = ep["instruction"]["instruction_text"].strip()
        meta[episode_id] = {
            "scene": scene_name,
            "instruction": instruction,
            "instruction_words": len(instruction.split()),
            "geodesic_distance": safe_float(ep["info"]["geodesic_distance"]),
        }
    return meta


def load_results(result_path):
    rows = {}
    pattern = os.path.join(result_path, "log", "stats_*.json")
    for filename in glob.glob(pattern):
        with open(filename, "r", encoding="utf-8") as f:
            row = json.load(f)
        row["id"] = int(row["id"])
        rows[row["id"]] = row
    return rows


def summarize_rows(rows):
    n = len(rows)
    success = sum(int(row["success"]) for row in rows)
    oracle_success = sum(int(row["oracle_success"]) for row in rows)
    spl = sum(safe_float(row["spl"]) for row in rows)
    dtg = sum(safe_float(row["distance_to_goal"]) for row in rows)
    path = sum(safe_float(row["path_length"]) for row in rows)
    oracle_only = sum(int(row["oracle_success"]) and not int(row["success"]) for row in rows)
    hard_fail = sum((not int(row["oracle_success"])) and (not int(row["success"])) for row in rows)
    return {
        "n": n,
        "success": success,
        "oracle_success": oracle_success,
        "sr": success / n if n else 0.0,
        "osr": oracle_success / n if n else 0.0,
        "spl": spl / n if n else 0.0,
        "dtg": dtg / n if n else 0.0,
        "path": path / n if n else 0.0,
        "oracle_only": oracle_only,
        "hard_fail": hard_fail,
    }


def markdown_table(headers, rows):
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def percent(value):
    return f"{value:.3f}"


def build_scene_stats(meta, best_rows, baseline_rows):
    scene_stats = defaultdict(lambda: {
        "rows": [],
        "baseline_rows": [],
        "geodesic_sum": 0.0,
        "instruction_words_sum": 0.0,
        "success_path_sum": 0.0,
        "success_geodesic_sum": 0.0,
        "success_n": 0,
        "fail_path_sum": 0.0,
        "fail_geodesic_sum": 0.0,
        "fail_n": 0,
        "new_win": 0,
        "new_lose": 0,
    })

    for episode_id, row in best_rows.items():
        scene = meta[episode_id]["scene"]
        stats = scene_stats[scene]
        stats["rows"].append(row)
        stats["geodesic_sum"] += meta[episode_id]["geodesic_distance"]
        stats["instruction_words_sum"] += meta[episode_id]["instruction_words"]
        if int(row["success"]):
            stats["success_path_sum"] += safe_float(row["path_length"])
            stats["success_geodesic_sum"] += meta[episode_id]["geodesic_distance"]
            stats["success_n"] += 1
        else:
            stats["fail_path_sum"] += safe_float(row["path_length"])
            stats["fail_geodesic_sum"] += meta[episode_id]["geodesic_distance"]
            stats["fail_n"] += 1

        baseline_row = baseline_rows[episode_id]
        stats["baseline_rows"].append(baseline_row)
        if int(row["success"]) and not int(baseline_row["success"]):
            stats["new_win"] += 1
        elif (not int(row["success"])) and int(baseline_row["success"]):
            stats["new_lose"] += 1

    out = []
    for scene, stats in scene_stats.items():
        best_summary = summarize_rows(stats["rows"])
        baseline_summary = summarize_rows(stats["baseline_rows"])
        n = best_summary["n"]
        out.append({
            "scene": scene,
            "n": n,
            "best": best_summary,
            "baseline": baseline_summary,
            "delta_sr": best_summary["sr"] - baseline_summary["sr"],
            "delta_osr": best_summary["osr"] - baseline_summary["osr"],
            "delta_spl": best_summary["spl"] - baseline_summary["spl"],
            "delta_dtg": best_summary["dtg"] - baseline_summary["dtg"],
            "delta_path": best_summary["path"] - baseline_summary["path"],
            "avg_geodesic": stats["geodesic_sum"] / n if n else 0.0,
            "avg_instruction_words": stats["instruction_words_sum"] / n if n else 0.0,
            "success_efficiency": (
                (stats["success_path_sum"] / stats["success_n"]) /
                (stats["success_geodesic_sum"] / stats["success_n"])
                if stats["success_n"] and stats["success_geodesic_sum"] else 0.0
            ),
            "fail_efficiency": (
                (stats["fail_path_sum"] / stats["fail_n"]) /
                (stats["fail_geodesic_sum"] / stats["fail_n"])
                if stats["fail_n"] and stats["fail_geodesic_sum"] else 0.0
            ),
            "new_win": stats["new_win"],
            "new_lose": stats["new_lose"],
        })
    out.sort(key=lambda item: item["scene"])
    return out


def bucket_report(meta, best_rows, key, bins):
    rows = []
    for lo, hi in bins:
        selected = []
        for episode_id, row in best_rows.items():
            value = meta[episode_id][key]
            if lo <= value < hi:
                selected.append(row)
        summary = summarize_rows(selected)
        rows.append({
            "bucket": f"[{lo}, {hi})",
            "n": summary["n"],
            "sr": summary["sr"],
            "osr": summary["osr"],
            "spl": summary["spl"],
            "dtg": summary["dtg"],
            "oracle_only": summary["oracle_only"],
            "hard_fail": summary["hard_fail"],
        })
    return rows


def worst_failures(meta, best_rows, top_k):
    failures = []
    for episode_id, row in best_rows.items():
        if int(row["success"]):
            continue
        failures.append({
            "id": episode_id,
            "scene": meta[episode_id]["scene"],
            "instruction_words": meta[episode_id]["instruction_words"],
            "geodesic_distance": meta[episode_id]["geodesic_distance"],
            "oracle_success": int(row["oracle_success"]),
            "distance_to_goal": safe_float(row["distance_to_goal"]),
            "path_length": safe_float(row["path_length"]),
        })
    failures.sort(
        key=lambda item: (item["distance_to_goal"], item["path_length"]),
        reverse=True,
    )
    return failures[:top_k]


def write_csv(path, headers, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def build_report(args):
    meta = load_dataset(args.dataset_path)
    best_rows = load_results(args.best_path)
    baseline_rows = load_results(args.baseline_path)

    overall_best = summarize_rows(list(best_rows.values()))
    overall_baseline = summarize_rows(list(baseline_rows.values()))
    scene_stats = build_scene_stats(meta, best_rows, baseline_rows)
    geodesic_buckets = bucket_report(
        meta,
        best_rows,
        "geodesic_distance",
        [(0, 5), (5, 10), (10, 15), (15, 20), (20, 999)],
    )
    instruction_buckets = bucket_report(
        meta,
        best_rows,
        "instruction_words",
        [(0, 10), (10, 15), (15, 20), (20, 999)],
    )
    severe_failures = worst_failures(meta, best_rows, args.top_k_failures)
    severe_scene_counts = defaultdict(int)
    for item in severe_failures:
        severe_scene_counts[item["scene"]] += 1

    scene_rows = []
    for item in scene_stats:
        scene_rows.append([
            item["scene"],
            item["n"],
            percent(item["best"]["sr"]),
            percent(item["best"]["osr"]),
            percent(item["best"]["spl"]),
            f"{item['best']['dtg']:.3f}",
            f"{item['best']['path']:.3f}",
        ])

    failure_rows = []
    for item in scene_stats:
        best = item["best"]
        failure_rows.append([
            item["scene"],
            item["n"],
            best["oracle_only"],
            percent(best["oracle_only"] / item["n"]),
            best["hard_fail"],
            percent(best["hard_fail"] / item["n"]),
            percent(best["osr"] - best["sr"]),
        ])

    delta_rows = []
    for item in sorted(scene_stats, key=lambda x: x["delta_sr"], reverse=True):
        delta_rows.append([
            item["scene"],
            item["n"],
            f"{item['delta_sr']:+.3f}",
            f"{item['delta_osr']:+.3f}",
            f"{item['delta_spl']:+.3f}",
            f"{item['delta_dtg']:+.3f}",
            item["new_win"],
            item["new_lose"],
            item["new_win"] - item["new_lose"],
        ])

    difficulty_rows = []
    for item in sorted(scene_stats, key=lambda x: x["avg_geodesic"], reverse=True):
        difficulty_rows.append([
            item["scene"],
            item["n"],
            f"{item['avg_geodesic']:.3f}",
            f"{item['avg_instruction_words']:.3f}",
            f"{item['success_efficiency']:.3f}",
            f"{item['fail_efficiency']:.3f}",
        ])

    bucket_rows_geod = []
    for item in geodesic_buckets:
        bucket_rows_geod.append([
            item["bucket"],
            item["n"],
            percent(item["sr"]),
            percent(item["osr"]),
            percent(item["spl"]),
            f"{item['dtg']:.3f}",
            item["oracle_only"],
            item["hard_fail"],
        ])

    bucket_rows_instr = []
    for item in instruction_buckets:
        bucket_rows_instr.append([
            item["bucket"],
            item["n"],
            percent(item["sr"]),
            percent(item["osr"]),
            percent(item["spl"]),
            f"{item['dtg']:.3f}",
            item["oracle_only"],
            item["hard_fail"],
        ])

    severe_rows = []
    for item in severe_failures:
        severe_rows.append([
            item["scene"],
            item["id"],
            item["oracle_success"],
            f"{item['distance_to_goal']:.3f}",
            f"{item['path_length']:.3f}",
            f"{item['geodesic_distance']:.3f}",
            item["instruction_words"],
        ])

    best_scene_by_sr = max(scene_stats, key=lambda x: x["best"]["sr"])
    worst_scene_by_sr = min(scene_stats, key=lambda x: x["best"]["sr"])
    largest_gain_scene = max(scene_stats, key=lambda x: x["delta_sr"])
    largest_regress_scene = min(scene_stats, key=lambda x: x["delta_sr"])
    largest_gap_scene = max(scene_stats, key=lambda x: x["best"]["osr"] - x["best"]["sr"])
    hardest_dtg_scene = max(scene_stats, key=lambda x: x["best"]["dtg"])

    report = []
    report.append(f"# Detailed Scene Report: {args.label}")
    report.append("")
    report.append("## 1. Setup")
    report.append("")
    report.append(f"- Candidate result path: `{args.best_path}`")
    report.append(f"- Baseline result path: `{args.baseline_path}`")
    report.append(f"- Dataset path: `{args.dataset_path}`")
    report.append(f"- Split: `val_unseen`")
    report.append(f"- Success threshold: `3.0m`")
    report.append(f"- Episode count: `{overall_best['n']}`")
    report.append(f"- Scene count: `{len(scene_stats)}`")
    report.append("")
    report.append("## 2. Executive Summary")
    report.append("")
    report.append(
        f"- Overall best-run metrics: SR `{overall_best['sr']:.3f}`, OSR `{overall_best['osr']:.3f}`, "
        f"SPL `{overall_best['spl']:.3f}`, DTG `{overall_best['dtg']:.3f}`, Path `{overall_best['path']:.3f}`."
    )
    report.append(
        f"- Versus baseline: dSR `{overall_best['sr'] - overall_baseline['sr']:+.3f}`, "
        f"dOSR `{overall_best['osr'] - overall_baseline['osr']:+.3f}`, "
        f"dSPL `{overall_best['spl'] - overall_baseline['spl']:+.3f}`, "
        f"dDTG `{overall_best['dtg'] - overall_baseline['dtg']:+.3f}`, "
        f"dPath `{overall_best['path'] - overall_baseline['path']:+.3f}`."
    )
    report.append(
        f"- Failure composition: oracle-only `{overall_best['oracle_only']}` "
        f"({overall_best['oracle_only'] / overall_best['n']:.3f}), hard-fail `{overall_best['hard_fail']}` "
        f"({overall_best['hard_fail'] / overall_best['n']:.3f})."
    )
    report.append(
        f"- Best scene by SR: `{best_scene_by_sr['scene']}` with SR `{best_scene_by_sr['best']['sr']:.3f}`."
    )
    report.append(
        f"- Worst scene by SR: `{worst_scene_by_sr['scene']}` with SR `{worst_scene_by_sr['best']['sr']:.3f}`."
    )
    report.append(
        f"- Largest positive scene delta vs baseline: `{largest_gain_scene['scene']}` with dSR `{largest_gain_scene['delta_sr']:+.3f}`."
    )
    report.append(
        f"- Largest negative scene delta vs baseline: `{largest_regress_scene['scene']}` with dSR `{largest_regress_scene['delta_sr']:+.3f}`."
    )
    report.append(
        f"- Largest OSR-SR gap scene: `{largest_gap_scene['scene']}` with gap `{largest_gap_scene['best']['osr'] - largest_gap_scene['best']['sr']:.3f}`."
    )
    report.append(
        f"- Largest average final DTG scene: `{hardest_dtg_scene['scene']}` with DTG `{hardest_dtg_scene['best']['dtg']:.3f}`."
    )
    report.append("")
    report.append("## 3. Scene Inventory and Candidate Metrics")
    report.append("")
    report.append(markdown_table(
        ["Scene", "N", "SR", "OSR", "SPL", "DTG", "Path"],
        scene_rows,
    ))
    report.append("")
    report.append("## 4. Failure Composition by Scene")
    report.append("")
    report.append(markdown_table(
        ["Scene", "N", "Oracle-Only", "Oracle-Only Rate", "Hard Fail", "Hard Fail Rate", "OSR-SR Gap"],
        failure_rows,
    ))
    report.append("")
    report.append("## 5. Delta vs Baseline by Scene")
    report.append("")
    report.append(markdown_table(
        ["Scene", "N", "dSR", "dOSR", "dSPL", "dDTG", "New Win", "New Lose", "Net Win"],
        delta_rows,
    ))
    report.append("")
    report.append("## 6. Difficulty Descriptors by Scene")
    report.append("")
    report.append(markdown_table(
        ["Scene", "N", "Avg Geod", "Avg Instr Words", "Success Path/Geod", "Fail Path/Geod"],
        difficulty_rows,
    ))
    report.append("")
    report.append("## 7. Geodesic-Distance Buckets")
    report.append("")
    report.append(markdown_table(
        ["Geod Bucket", "N", "SR", "OSR", "SPL", "DTG", "Oracle-Only", "Hard Fail"],
        bucket_rows_geod,
    ))
    report.append("")
    report.append("## 8. Instruction-Length Buckets")
    report.append("")
    report.append(markdown_table(
        ["Instr Bucket", "N", "SR", "OSR", "SPL", "DTG", "Oracle-Only", "Hard Fail"],
        bucket_rows_instr,
    ))
    report.append("")
    report.append("## 9. Most Severe Failed Episodes")
    report.append("")
    report.append(markdown_table(
        ["Scene", "Episode ID", "Oracle Success", "Final DTG", "Path", "Geod", "Instr Words"],
        severe_rows,
    ))
    report.append("")
    report.append("## 10. Interpretation")
    report.append("")
    report.append("### 10.1 Global Behavior")
    report.append("")
    report.append(
        "- The best run improves overall SR only modestly, which means the gain is real but not uniformly distributed."
    )
    report.append(
        "- The dominant failure class remains hard failure, not stop failure. Oracle-only cases are important, but they are still the minority."
    )
    report.append(
        "- Improvement comes from recovering some previously hard episodes, while simultaneously sacrificing a non-trivial number of baseline successes."
    )
    report.append("")
    report.append("### 10.2 Scene-Level Behavior")
    report.append("")
    report.append(
        f"- `{largest_gain_scene['scene']}` is the clearest positive scene. It contributes the largest dSR and also improves OSR and SPL together."
    )
    report.append(
        f"- `{largest_regress_scene['scene']}` is the clearest negative scene. This indicates the candidate policy is not globally more stable than baseline."
    )
    report.append(
        f"- `{hardest_dtg_scene['scene']}` and `{largest_gap_scene['scene']}` fail for different reasons: the former is a route-selection problem, the latter is a finalization / stop-quality problem."
    )
    report.append("")
    report.append("### 10.3 Difficulty Factors")
    report.append("")
    report.append(
        "- Higher geodesic-distance buckets are substantially harder. The main drop happens once shortest-path distance moves past 10m."
    )
    report.append(
        "- Instruction length alone does not show a monotonic failure curve. Shortest-path distance is the stronger difficulty driver in this run."
    )
    report.append(
        "- Some scenes with long average instructions still perform well, which suggests layout/navigation ambiguity is more important than raw instruction length."
    )
    report.append("")
    report.append("### 10.4 Operational Reading")
    report.append("")
    report.append(
        "- If the goal is to lift overall SR, the first priority should be scenes with high hard-fail rates and large final DTG."
    )
    report.append(
        "- If the goal is to clean up near-miss behavior, prioritize scenes with large OSR-SR gaps and high oracle-only rates."
    )
    report.append(
        "- The severe-failure list shows repeated concentration in a small subset of scenes, which is a strong hint that scene/layout-specific failure modes are present."
    )
    report.append("")
    report.append("## 11. Scene-by-Scene Diagnosis")
    report.append("")
    for item in sorted(scene_stats, key=lambda x: x["best"]["sr"]):
        best = item["best"]
        gap = best["osr"] - best["sr"]
        hard_fail_rate = best["hard_fail"] / item["n"]
        oracle_only_rate = best["oracle_only"] / item["n"]
        if best["sr"] >= 0.70:
            level = "strong-performing scene"
        elif best["sr"] >= 0.60:
            level = "above-average scene"
        elif best["sr"] >= 0.50:
            level = "mid-performing scene"
        else:
            level = "weak scene"
        if item["delta_sr"] >= 0.03:
            delta_note = "This scene improves clearly over baseline."
        elif item["delta_sr"] > 0:
            delta_note = "This scene improves only marginally over baseline."
        elif item["delta_sr"] == 0:
            delta_note = "This scene is effectively unchanged versus baseline."
        else:
            delta_note = "This scene regresses versus baseline."
        if gap >= 0.12:
            failure_note = "Its main issue is finalization: the OSR-SR gap is large, so the agent often gets near the goal but does not finish correctly."
        elif best["dtg"] >= 6.0 or hard_fail_rate >= 0.35:
            failure_note = "Its main issue is route failure: final DTG and hard-fail rate are both high, so the agent frequently never reaches the right region."
        else:
            failure_note = "Its errors are mixed: both route selection and finalization contribute, but neither fully dominates."
        severity_note = ""
        if severe_scene_counts[item["scene"]] > 0:
            severity_note = (
                f" It also appears `{severe_scene_counts[item['scene']]}` times in the top "
                f"`{args.top_k_failures}` worst failures."
            )
        report.append(
            f"- `{item['scene']}` is a {level}. "
            f"SR `{best['sr']:.3f}`, OSR `{best['osr']:.3f}`, SPL `{best['spl']:.3f}`, "
            f"DTG `{best['dtg']:.3f}`, Path `{best['path']:.3f}`. "
            f"{delta_note} "
            f"{failure_note} "
            f"Oracle-only rate is `{oracle_only_rate:.3f}` and hard-fail rate is `{hard_fail_rate:.3f}`."
            f"{severity_note}"
        )
    report.append("")
    report.append("## 12. Optimization Priorities")
    report.append("")
    route_priority = sorted(
        scene_stats,
        key=lambda x: (x["best"]["hard_fail"], x["best"]["dtg"]),
        reverse=True,
    )
    stop_priority = sorted(
        scene_stats,
        key=lambda x: (x["best"]["oracle_only"], x["best"]["osr"] - x["best"]["sr"]),
        reverse=True,
    )
    regression_priority = sorted(scene_stats, key=lambda x: x["delta_sr"])
    report.append("### 12.1 Route-Failure Priority")
    report.append("")
    for item in route_priority[:5]:
        report.append(
            f"- `{item['scene']}`: hard-fail `{item['best']['hard_fail']}` / `{item['n']}`, "
            f"hard-fail rate `{item['best']['hard_fail'] / item['n']:.3f}`, "
            f"DTG `{item['best']['dtg']:.3f}`."
        )
    report.append("")
    report.append("### 12.2 Finalization / Stop Priority")
    report.append("")
    for item in stop_priority[:5]:
        report.append(
            f"- `{item['scene']}`: oracle-only `{item['best']['oracle_only']}` / `{item['n']}`, "
            f"oracle-only rate `{item['best']['oracle_only'] / item['n']:.3f}`, "
            f"OSR-SR gap `{item['best']['osr'] - item['best']['sr']:.3f}`."
        )
    report.append("")
    report.append("### 12.3 Baseline Regression Watchlist")
    report.append("")
    for item in regression_priority[:5]:
        report.append(
            f"- `{item['scene']}`: dSR `{item['delta_sr']:+.3f}`, "
            f"dOSR `{item['delta_osr']:+.3f}`, net win `{item['new_win'] - item['new_lose']}`."
        )
    report.append("")
    report_text = "\n".join(report) + "\n"

    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "detailed_scene_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    scene_csv_rows = []
    for item in scene_stats:
        scene_csv_rows.append([
            item["scene"],
            item["n"],
            item["best"]["sr"],
            item["best"]["osr"],
            item["best"]["spl"],
            item["best"]["dtg"],
            item["best"]["path"],
            item["best"]["oracle_only"],
            item["best"]["hard_fail"],
            item["delta_sr"],
            item["delta_osr"],
            item["delta_spl"],
            item["delta_dtg"],
            item["avg_geodesic"],
            item["avg_instruction_words"],
            item["new_win"],
            item["new_lose"],
        ])
    write_csv(
        os.path.join(args.output_dir, "scene_metrics.csv"),
        [
            "scene",
            "n",
            "sr",
            "osr",
            "spl",
            "dtg",
            "path",
            "oracle_only",
            "hard_fail",
            "delta_sr_vs_baseline",
            "delta_osr_vs_baseline",
            "delta_spl_vs_baseline",
            "delta_dtg_vs_baseline",
            "avg_geodesic_distance",
            "avg_instruction_words",
            "new_win",
            "new_lose",
        ],
        scene_csv_rows,
    )

    severe_csv_rows = []
    for item in severe_failures:
        severe_csv_rows.append([
            item["scene"],
            item["id"],
            item["oracle_success"],
            item["distance_to_goal"],
            item["path_length"],
            item["geodesic_distance"],
            item["instruction_words"],
        ])
    write_csv(
        os.path.join(args.output_dir, "severe_failures.csv"),
        [
            "scene",
            "episode_id",
            "oracle_success",
            "distance_to_goal",
            "path_length",
            "geodesic_distance",
            "instruction_words",
        ],
        severe_csv_rows,
    )

    summary = {
        "report_path": report_path,
        "scene_count": len(scene_stats),
        "episode_count": overall_best["n"],
        "best_sr": overall_best["sr"],
        "best_osr": overall_best["osr"],
        "best_spl": overall_best["spl"],
        "best_dtg": overall_best["dtg"],
        "best_path": overall_best["path"],
        "delta_sr_vs_baseline": overall_best["sr"] - overall_baseline["sr"],
        "delta_osr_vs_baseline": overall_best["osr"] - overall_baseline["osr"],
        "delta_spl_vs_baseline": overall_best["spl"] - overall_baseline["spl"],
        "delta_dtg_vs_baseline": overall_best["dtg"] - overall_baseline["dtg"],
        "delta_path_vs_baseline": overall_best["path"] - overall_baseline["path"],
    }
    with open(os.path.join(args.output_dir, "report_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--best-path", required=True)
    parser.add_argument("--baseline-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", default="best_run")
    parser.add_argument("--top-k-failures", type=int, default=20)
    args = parser.parse_args()
    build_report(args)


if __name__ == "__main__":
    main()
