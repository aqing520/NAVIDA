import argparse
import glob
import json
import math
import os


def safe_float(value):
    value = float(value)
    if math.isinf(value) or math.isnan(value):
        return 0.0
    return value


def load_rows(path):
    rows = []
    for filename in glob.glob(os.path.join(path, "log", "stats_*.json")):
        with open(filename, "r", encoding="utf-8") as f:
            row = json.load(f)
        row["id"] = int(row["id"])
        rows.append(row)
    rows.sort(key=lambda item: item["id"])
    return rows


def row_brief(row):
    return {
        "id": int(row["id"]),
        "success": int(row["success"]),
        "oracle_success": int(row["oracle_success"]),
        "distance_to_goal": safe_float(row["distance_to_goal"]),
        "path_length": safe_float(row["path_length"]),
        "spl": safe_float(row["spl"]),
        "ndtw": safe_float(row.get("ndtw", 0.0)),
    }


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_id_list(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{int(row['id'])}\n")


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def count_failures(rows):
    return sum(not int(row["success"]) for row in rows)


def main():
    parser = argparse.ArgumentParser(
        description="Export failure episode IDs for a candidate run."
    )
    parser.add_argument("--path", required=True, help="candidate result directory")
    parser.add_argument(
        "--baseline-path",
        default=None,
        help="optional baseline result directory for flipped sample export",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="directory to write exported IDs and details",
    )
    args = parser.parse_args()

    candidate_rows = load_rows(args.path)
    candidate_by_id = {row["id"]: row for row in candidate_rows}

    oracle_only_rows = [
        row for row in candidate_rows
        if int(row["oracle_success"]) == 1 and int(row["success"]) == 0
    ]
    hard_fail_rows = [
        row for row in candidate_rows
        if int(row["oracle_success"]) == 0 and int(row["success"]) == 0
    ]

    baseline_rows = []
    new_success_rows = []
    new_fail_rows = []
    aligned_n = 0
    if args.baseline_path:
        baseline_rows = load_rows(args.baseline_path)
        baseline_by_id = {row["id"]: row for row in baseline_rows}
        aligned_ids = sorted(set(candidate_by_id) & set(baseline_by_id))
        aligned_n = len(aligned_ids)
        new_success_rows = []
        new_fail_rows = []
        for episode_id in aligned_ids:
            candidate_row = candidate_by_id[episode_id]
            baseline_row = baseline_by_id[episode_id]
            if int(candidate_row["success"]) == 1 and int(baseline_row["success"]) == 0:
                new_success_rows.append({
                    "id": episode_id,
                    "candidate": row_brief(candidate_row),
                    "baseline": row_brief(baseline_row),
                })
            elif int(candidate_row["success"]) == 0 and int(baseline_row["success"]) == 1:
                new_fail_rows.append({
                    "id": episode_id,
                    "candidate": row_brief(candidate_row),
                    "baseline": row_brief(baseline_row),
                })

    ensure_dir(args.output_dir)

    write_id_list(
        os.path.join(args.output_dir, "oracle_only_ids.txt"),
        oracle_only_rows,
    )
    write_id_list(
        os.path.join(args.output_dir, "hard_fail_ids.txt"),
        hard_fail_rows,
    )
    write_json(
        os.path.join(args.output_dir, "oracle_only_details.json"),
        [row_brief(row) for row in oracle_only_rows],
    )
    write_json(
        os.path.join(args.output_dir, "hard_fail_details.json"),
        [row_brief(row) for row in hard_fail_rows],
    )

    if args.baseline_path:
        write_id_list(
            os.path.join(args.output_dir, "flip_new_success_ids.txt"),
            new_success_rows,
        )
        write_id_list(
            os.path.join(args.output_dir, "flip_new_fail_ids.txt"),
            new_fail_rows,
        )
        write_json(
            os.path.join(args.output_dir, "flip_new_success_details.json"),
            new_success_rows,
        )
        write_json(
            os.path.join(args.output_dir, "flip_new_fail_details.json"),
            new_fail_rows,
        )

    summary = {
        "candidate_path": args.path,
        "baseline_path": args.baseline_path,
        "candidate_episode_count": len(candidate_rows),
        "candidate_fail_count": count_failures(candidate_rows),
        "oracle_only_count": len(oracle_only_rows),
        "hard_fail_count": len(hard_fail_rows),
        "aligned_episode_count": aligned_n,
        "flip_new_success_count": len(new_success_rows),
        "flip_new_fail_count": len(new_fail_rows),
        "output_dir": args.output_dir,
    }
    write_json(os.path.join(args.output_dir, "summary.json"), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
