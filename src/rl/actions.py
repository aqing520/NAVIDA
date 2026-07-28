import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParsedAction:
    action_id: int
    text: str
    valid: bool


@dataclass
class ParsedActionChunk:
    actions: list[ParsedAction]
    text: str
    valid: bool


def action_id_to_text(action_id: int, forward_distance: int = 25, turn_angle: int = 15) -> str:
    if action_id == 0:
        return "stop"
    if action_id == 1:
        return f"forward {forward_distance} cm"
    if action_id == 2:
        return f"turn left {turn_angle} degree"
    if action_id == 3:
        return f"turn right {turn_angle} degree"
    raise ValueError(f"Invalid action id: {action_id}")


def parse_action_text(output: str, forward_distance: int = 25, turn_angle: int = 15) -> ParsedAction:
    chunk = parse_action_chunk_text(output, forward_distance, turn_angle, max_actions=1)
    return chunk.actions[0]


def parse_action_chunk_text(
    output: str,
    forward_distance: int = 25,
    turn_angle: int = 15,
    max_actions: int = 2,
    max_repeat: int = 3,
) -> ParsedActionChunk:
    action_texts = _action_texts(output)
    parsed_actions: list[ParsedAction] = []
    valid = True

    for action_text in action_texts[:max_actions]:
        sub_actions = _parse_one_action(action_text, forward_distance, turn_angle, max_repeat)
        if not sub_actions:
            valid = False
            sub_actions = [ParsedAction(action_id=1, text=action_id_to_text(1, forward_distance, turn_angle), valid=False)]
        parsed_actions.extend(sub_actions)

    if not parsed_actions:
        valid = False
        parsed_actions = [ParsedAction(action_id=1, text=action_id_to_text(1, forward_distance, turn_angle), valid=False)]

    return ParsedActionChunk(
        actions=parsed_actions,
        text=", ".join(action.text for action in parsed_actions),
        valid=valid and all(action.valid for action in parsed_actions),
    )


def _parse_one_action(
    action_text: str,
    forward_distance: int,
    turn_angle: int,
    max_repeat: int,
) -> list[ParsedAction]:
    action_id, numeric = _extract_action_and_value(action_text)
    if action_id is None:
        return []

    if action_id == 0:
        return [ParsedAction(action_id=0, text="stop", valid=True)]
    if action_id == 1:
        value = numeric if numeric is not None else forward_distance
        steps = round(value / forward_distance)
        if steps <= 0:
            return []
        steps = min(max_repeat, steps)
        return [
            ParsedAction(action_id=1, text=action_id_to_text(1, forward_distance, turn_angle), valid=True)
            for _ in range(steps)
        ]
    if action_id == 2:
        value = numeric if numeric is not None else turn_angle
        steps = round(value / turn_angle)
        if steps <= 0:
            return []
        steps = min(max_repeat, steps)
        return [
            ParsedAction(action_id=2, text=action_id_to_text(2, forward_distance, turn_angle), valid=True)
            for _ in range(steps)
        ]
    if action_id == 3:
        value = numeric if numeric is not None else turn_angle
        steps = round(value / turn_angle)
        if steps <= 0:
            return []
        steps = min(max_repeat, steps)
        return [
            ParsedAction(action_id=3, text=action_id_to_text(3, forward_distance, turn_angle), valid=True)
            for _ in range(steps)
        ]
    raise ValueError(f"Invalid parsed action id: {action_id}")


def _action_texts(output: str) -> list[str]:
    match = re.search(r"<answer>(.*?)</answer>", output, flags=re.DOTALL)
    output = match.group(1) if match else output
    return [item.strip() for item in re.split(r"\s*,\s*", output.strip()) if item.strip()]


def _extract_action_and_value(text: str) -> tuple[Optional[int], Optional[float]]:
    text = text.lower()
    if "stop" in text:
        return 0, None
    match = re.search(r"-?\d+", text)
    numeric = float(match.group()) if match is not None else None
    if "forward" in text:
        return 1, numeric
    if "left" in text:
        return 2, numeric
    if "right" in text:
        return 3, numeric
    return None, None
