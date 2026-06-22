import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParsedAction:
    action_id: int
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
    action_text = _first_action_text(output)
    action_id, numeric = _extract_action_and_value(action_text)
    if action_id is None:
        return ParsedAction(action_id=1, text=action_id_to_text(1, forward_distance, turn_angle), valid=False)

    if action_id == 0:
        return ParsedAction(action_id=0, text="stop", valid=True)
    if action_id == 1:
        steps = max(1, round((numeric or forward_distance) / forward_distance))
        steps = min(3, steps)
        return ParsedAction(
            action_id=1,
            text=action_id_to_text(1, steps * forward_distance, turn_angle),
            valid=True,
        )
    if action_id == 2:
        steps = max(1, round((numeric or turn_angle) / turn_angle))
        steps = min(3, steps)
        return ParsedAction(
            action_id=2,
            text=action_id_to_text(2, forward_distance, steps * turn_angle),
            valid=True,
        )
    if action_id == 3:
        steps = max(1, round((numeric or turn_angle) / turn_angle))
        steps = min(3, steps)
        return ParsedAction(
            action_id=3,
            text=action_id_to_text(3, forward_distance, steps * turn_angle),
            valid=True,
        )
    raise ValueError(f"Invalid parsed action id: {action_id}")


def _first_action_text(output: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", output, flags=re.DOTALL)
    output = match.group(1) if match else output
    parts = [item.strip() for item in re.split(r"\s*,\s*", output.strip()) if item.strip()]
    return parts[0] if parts else output.strip()


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

