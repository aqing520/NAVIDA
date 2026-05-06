"""
Build high-confidence correction training data from failed rollouts.

Key differences from v5/v6/v7:
- Detect deviation geometrically on the recorded rollout trace.
- Recompute recovery actions from the agent's actual pose with
  ShortestPathFollower instead of trusting recorded oracle_action labels.
- Keep only short, high-confidence recoveries that make measurable progress.
"""

import os
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["GLOG_minloglevel"] = "2"

import json
import gzip
import argparse
import random
import re
from collections import Counter, defaultdict

import numpy as np
from tqdm import tqdm

import habitat
from habitat_baselines.config.default import get_config
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

from habitat_extensions import measures, task  # noqa: F401


FORWARD_DISTANCE = 25
TURN_ANGLE = 15

VLN_PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. "
    "You have been given a video of historical observations and an image of the current observation. "
    "Your assigned task is: '{}'. Analyze this series of images to decide your next move, "
    "which could involve turning left or right by a specific degree or moving forward a certain distance."
)


def action_id_to_str(action_id):
    if action_id == 0:
        return "stop"
    if action_id == 1:
        return f"forward {FORWARD_DISTANCE} cm"
    if action_id == 2:
        return f"turn left {TURN_ANGLE} degree"
    if action_id == 3:
        return f"turn right {TURN_ANGLE} degree"
    raise ValueError(f"Invalid action ID: {action_id}")


def combine(action1, action2):
    idx = action1.rfind(", ")
    subaction0 = action1[: idx + 1] + " " if action1[: idx + 1] != "" else action1[: idx + 1]
    subaction1 = action1[idx + 1 :]
    match1 = int(re.search(r"-?\d+", subaction1).group())
    match2 = int(re.search(r"-?\d+", action2).group())
    if "forward" in subaction1:
        if match1 + match2 <= 3 * FORWARD_DISTANCE:
            return f"{subaction0}forward {match1 + match2} cm"
        return None
    if "turn left" in subaction1:
        if match1 + match2 <= 3 * TURN_ANGLE:
            return f"{subaction0}turn left {match1 + match2} degree"
        return None
    if "turn right" in subaction1:
        if match1 + match2 <= 3 * TURN_ANGLE:
            return f"{subaction0}turn right {match1 + match2} degree"
        return None
    raise ValueError(f"Invalid action: {action1}")


def build_action_chunk(action_ids):
    if not action_ids:
        return None, 0

    first_action = action_ids[0]
    if first_action == 0:
        return "stop", 1

    chunk = action_id_to_str(first_action)
    last_action = first_action
    actions_used = 1

    for i in range(1, len(action_ids)):
        next_action = action_ids[i]
        if next_action == 0:
            break

        next_action_str = action_id_to_str(next_action)
        if random.random() <= 0.7 and next_action == last_action:
            merged = combine(chunk, next_action_str)
            if merged is not None:
                chunk = merged
                last_action = next_action
                actions_used += 1
                continue

        if chunk.count(",") < 2:
            chunk += ", " + next_action_str
            last_action = next_action
            actions_used += 1
        else:
            break

    return chunk, actions_used


def interpolate_path(waypoints, interval=0.25):
    if len(waypoints) < 2:
        return np.array(waypoints)

    waypoints = [np.array(p[:3]) for p in waypoints]
    dense = [waypoints[0]]
    for i in range(len(waypoints) - 1):
        seg = waypoints[i + 1] - waypoints[i]
        seg_len = np.linalg.norm(seg)
        if seg_len < 1e-6:
            continue
        n_points = max(1, int(seg_len / interval))
        for j in range(1, n_points + 1):
            t = j / n_points
            dense.append(waypoints[i] + t * seg)
    return np.array(dense)


def point_to_path_distance(point, dense_path):
    point = np.array(point[:3])
    diffs = dense_path - point
    dists = np.linalg.norm(diffs, axis=1)
    return float(np.min(dists))


def load_reference_paths(annotations_path):
    with gzip.open(annotations_path, "rt") as f:
        data = json.load(f)
    ref_paths = {}
    for ep in data["episodes"]:
        ref_paths[str(ep["episode_id"])] = ep.get("reference_path", [])
    return ref_paths


def load_traces(rollout_dir):
    import glob

    traces = {}
    trace_dir = os.path.join(rollout_dir, "traces")
    for trace_file in glob.glob(os.path.join(trace_dir, "*.jsonl")):
        episode_id = os.path.splitext(os.path.basename(trace_file))[0]
        records = []
        with open(trace_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        if records:
            traces[episode_id] = records
    return traces


def load_summaries(rollout_dir):
    import glob

    summaries = {}
    summary_dir = os.path.join(rollout_dir, "summaries")
    for path in glob.glob(os.path.join(summary_dir, "*.json")):
        with open(path, "r", encoding="utf-8") as f:
            item = json.load(f)
        summaries[str(item["episode_id"])] = item
    return summaries


def get_context_frames(records, deviation_idx, max_history):
    start = max(0, deviation_idx - max_history + 1)
    frames = []
    for i in range(start, deviation_idx + 1):
        frame_path = records[i].get("frame_path", "")
        abs_path = os.path.abspath(frame_path)
        if os.path.exists(abs_path):
            frames.append(abs_path)
    return frames


def find_deviation_record(records, dense_path, threshold):
    for idx, record in enumerate(records):
        if not record.get("is_decision_step", False):
            continue
        agent_pos = record.get("agent_position")
        if agent_pos is None:
            continue
        deviation_distance = point_to_path_distance(agent_pos, dense_path)
        if deviation_distance > threshold:
            return idx, record, deviation_distance
    return None, None, None


def replay_recovery(env, episode, start_position, start_rotation, goal_position,
                    max_recovery_steps, goal_radius):
    env.current_episode = episode
    env.reset()

    ok = env._sim.set_agent_state(start_position, start_rotation)
    if ok is False:
        return None

    follower = ShortestPathFollower(env._sim, goal_radius=goal_radius, return_one_hot=False)
    actions = []
    dtgs = []
    oracle_stop = False

    for _ in range(max_recovery_steps):
        action = follower.get_next_action(goal_position)
        if action is None:
            break
        action = int(action)
        actions.append(action)
        if action == 0:
            oracle_stop = True
            break
        observations = env.step(action)
        info = env.get_metrics()
        dtgs.append(float(info.get("distance_to_goal", 0.0)))
        if info.get("success", 0):
            break
        # Keep the observation alive to ensure the simulator state is updated.
        _ = observations

    return {
        "actions": actions,
        "dtgs": dtgs,
        "oracle_stop": oracle_stop,
    }


def build_correction_samples(
    rollout_dir,
    annotations_path,
    exp_config,
    threshold=1.5,
    max_history=8,
    max_recovery_steps=6,
    min_deviation_distance=1.5,
    max_deviation_distance=3.5,
    min_goal_distance=2.0,
    max_goal_distance=20.0,
    min_progress=0.5,
    goal_radius=3.0,
):
    ref_paths = load_reference_paths(annotations_path)
    traces = load_traces(rollout_dir)
    summaries = load_summaries(rollout_dir)

    failed_episode_ids = {
        episode_id for episode_id, summary in summaries.items()
        if not summary.get("final_success", 0)
    }

    dense_paths = {}
    for episode_id, ref_path in ref_paths.items():
        if ref_path and len(ref_path) >= 2:
            dense_paths[episode_id] = interpolate_path(ref_path, interval=0.25)

    env_config = get_config(exp_config)
    dataset = habitat.datasets.make_dataset(
        id_dataset=env_config.habitat.dataset.type,
        config=env_config.habitat.dataset,
    )
    episode_map = {str(ep.episode_id): ep for ep in dataset.episodes}
    env = habitat.Env(env_config.habitat, dataset)

    samples = []
    reason_counts = Counter()
    first_action_counts = Counter()
    source_counts = Counter()

    ordered_failed_episode_ids = sorted(
        failed_episode_ids,
        key=lambda episode_id: (
            episode_map[episode_id].scene_id if episode_id in episode_map else "",
            int(episode_id),
        ),
    )

    for episode_id in tqdm(ordered_failed_episode_ids, desc="Building v8 correction data"):
        records = traces.get(episode_id)
        dense_path = dense_paths.get(episode_id)
        episode = episode_map.get(episode_id)

        if not records:
            reason_counts["missing_trace"] += 1
            continue
        if dense_path is None:
            reason_counts["missing_reference_path"] += 1
            continue
        if episode is None:
            reason_counts["missing_episode"] += 1
            continue

        deviation_idx, deviation_record, deviation_distance = find_deviation_record(
            records, dense_path, threshold
        )
        if deviation_record is None:
            reason_counts["no_deviation"] += 1
            continue

        distance_to_goal = float(deviation_record.get("distance_to_goal", 0.0))
        if deviation_distance < min_deviation_distance:
            reason_counts["deviation_too_small"] += 1
            continue
        if deviation_distance > max_deviation_distance:
            reason_counts["deviation_too_large"] += 1
            continue
        if distance_to_goal < min_goal_distance:
            reason_counts["goal_too_close"] += 1
            continue
        if distance_to_goal > max_goal_distance:
            reason_counts["goal_too_far"] += 1
            continue

        context_frames = get_context_frames(records, deviation_idx, max_history)
        if not context_frames:
            reason_counts["missing_context_frames"] += 1
            continue

        start_position = deviation_record.get("agent_position")
        start_rotation = deviation_record.get("agent_rotation")
        goal_position = deviation_record.get("goal_position")
        instruction = deviation_record.get("instruction", "")

        if not start_position or not start_rotation or not goal_position:
            reason_counts["missing_pose_or_goal"] += 1
            continue
        if not instruction:
            reason_counts["missing_instruction"] += 1
            continue

        replay = replay_recovery(
            env=env,
            episode=episode,
            start_position=start_position,
            start_rotation=start_rotation,
            goal_position=goal_position,
            max_recovery_steps=max_recovery_steps,
            goal_radius=goal_radius,
        )
        if replay is None:
            reason_counts["set_state_failed"] += 1
            continue

        recovery_actions = replay["actions"]
        if not recovery_actions:
            reason_counts["empty_recovery"] += 1
            continue
        if recovery_actions[0] == 0:
            reason_counts["starts_with_stop"] += 1
            continue

        min_replayed_dtg = min(replay["dtgs"]) if replay["dtgs"] else distance_to_goal
        if min_replayed_dtg > distance_to_goal - min_progress and not replay["oracle_stop"]:
            reason_counts["insufficient_progress"] += 1
            continue

        if len(recovery_actions) > max_recovery_steps:
            reason_counts["recovery_too_long"] += 1
            continue

        chunk_str, actions_used = build_action_chunk(recovery_actions)
        if chunk_str is None:
            reason_counts["chunk_build_failed"] += 1
            continue

        first_action = chunk_str.split(",")[0].strip().split()[0]
        first_action_counts[first_action] += 1
        source_counts["correctnav_replayed_spf"] += 1

        sample = {
            "system": "You are a helpful assistant.",
            "conversations": [
                {
                    "from": "user",
                    "value": VLN_PROMPT_TEMPLATE.format(instruction),
                    "image": context_frames,
                },
                {
                    "from": "assistant",
                    "value": chunk_str,
                },
            ],
            "action_history": [],
            "task type": "vln",
            "episode_id": str(episode_id),
            "source": "correctnav_replayed_spf",
            "deviation_step": int(deviation_record["step"]),
            "deviation_distance": float(deviation_distance),
            "distance_to_goal": float(distance_to_goal),
            "recovery_actions": recovery_actions,
            "recovery_actions_used_in_chunk": int(actions_used),
            "recovery_min_distance_to_goal": float(min_replayed_dtg),
        }
        samples.append(sample)

    env.close()

    stats = {
        "rollout_dir": rollout_dir,
        "annotations_path": annotations_path,
        "exp_config": exp_config,
        "total_failed_episodes": len(failed_episode_ids),
        "built_samples": len(samples),
        "skip_reasons": dict(reason_counts),
        "first_action_distribution": dict(first_action_counts),
        "source_distribution": dict(source_counts),
        "threshold": threshold,
        "max_recovery_steps": max_recovery_steps,
        "min_deviation_distance": min_deviation_distance,
        "max_deviation_distance": max_deviation_distance,
        "min_goal_distance": min_goal_distance,
        "max_goal_distance": max_goal_distance,
        "min_progress": min_progress,
        "goal_radius": goal_radius,
    }
    return samples, stats


def main():
    parser = argparse.ArgumentParser(
        description="Build high-confidence CorrectNAV-style correction data from rollout traces."
    )
    parser.add_argument("--rollout-dir", type=str, required=True)
    parser.add_argument("--annotations-path", type=str, required=True)
    parser.add_argument("--exp-config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=1.5)
    parser.add_argument("--max-history", type=int, default=8)
    parser.add_argument("--max-recovery-steps", type=int, default=6)
    parser.add_argument("--min-deviation-distance", type=float, default=1.5)
    parser.add_argument("--max-deviation-distance", type=float, default=3.5)
    parser.add_argument("--min-goal-distance", type=float, default=2.0)
    parser.add_argument("--max-goal-distance", type=float, default=20.0)
    parser.add_argument("--min-progress", type=float, default=0.5)
    parser.add_argument("--goal-radius", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    samples, stats = build_correction_samples(
        rollout_dir=args.rollout_dir,
        annotations_path=args.annotations_path,
        exp_config=args.exp_config,
        threshold=args.threshold,
        max_history=args.max_history,
        max_recovery_steps=args.max_recovery_steps,
        min_deviation_distance=args.min_deviation_distance,
        max_deviation_distance=args.max_deviation_distance,
        min_goal_distance=args.min_goal_distance,
        max_goal_distance=args.max_goal_distance,
        min_progress=args.min_progress,
        goal_radius=args.goal_radius,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    stats_path = args.output.replace(".jsonl", "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Built {len(samples)} samples")
    print(f"Output: {args.output}")
    print(f"Stats: {stats_path}")
    print(f"Skip reasons: {stats['skip_reasons']}")
    print(f"First action distribution: {stats['first_action_distribution']}")


if __name__ == "__main__":
    main()
