from dataclasses import dataclass
from typing import Any


@dataclass
class RewardConfig:
    distance_delta_scale: float = 1.0
    success_bonus: float = 5.0
    wrong_stop_penalty: float = -3.0
    invalid_action_penalty: float = -0.5
    collision_penalty: float = -0.2
    step_penalty: float = -0.01
    goal_distance: float = 3.0


def compute_step_reward(
    prev_metrics: dict[str, Any],
    next_metrics: dict[str, Any],
    action_id: int,
    action_valid: bool,
    cfg: RewardConfig,
) -> float:
    prev_dtg = float(prev_metrics.get("distance_to_goal", 0.0))
    next_dtg = float(next_metrics.get("distance_to_goal", prev_dtg))
    reward = cfg.distance_delta_scale * (prev_dtg - next_dtg)
    reward += cfg.step_penalty

    if not action_valid:
        reward += cfg.invalid_action_penalty

    if _is_collision(next_metrics):
        reward += cfg.collision_penalty

    if action_id == 0:
        if float(next_metrics.get("success", 0.0)) > 0.0 or next_dtg <= cfg.goal_distance:
            reward += cfg.success_bonus
        else:
            reward += cfg.wrong_stop_penalty
    return float(reward)


def _is_collision(metrics: dict[str, Any]) -> bool:
    collisions = metrics.get("collisions")
    if isinstance(collisions, dict):
        return bool(collisions.get("is_collision", False))
    return bool(collisions) if collisions is not None else False

