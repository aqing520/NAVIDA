import base64
import io
from typing import List

from PIL import Image


SYSTEM_PROMPT = "You are a helpful assistant."

BASE_PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. "
    "You have been given a video of historical observations and an image of the current observation. "
    "Your assigned task is: '{}'. Analyze this series of images to decide your next move, "
    "which could involve turning left or right by a specific degree or moving forward a certain distance."
)

RL_PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. "
    "You have been given a video of historical observations and an image of the current observation. "
    "Your assigned task is: '{}'. Analyze this series of images to decide your next move. "
    "Available actions are: stop; forward 25 cm; turn left 15 degree; turn right 15 degree. "
    "Respond with exactly one action in one of these formats: stop, forward 25 cm, "
    "turn left 15 degree, turn right 15 degree."
)


def encode_image_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def uniform_sample_with_ends(data: List[Image.Image], n: int) -> List[Image.Image]:
    if len(data) <= n:
        return data
    indices = [round(i * (len(data) - 1) / (n - 1)) for i in range(n)]
    return [data[i] for i in indices]


def build_vln_prompt_messages(
    instruction: str,
    rgb_history: List[Image.Image],
    max_history_images: int = 8,
    prompt_template: str = RL_PROMPT_TEMPLATE,
) -> list[dict]:
    if not rgb_history:
        raise ValueError("rgb_history must contain at least one frame")

    content = [
        {
            "type": "text",
            "text": "Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations",
        }
    ]
    if len(rgb_history) > 1:
        history = uniform_sample_with_ends(rgb_history[:-1], max_history_images)
        content.extend(_image_items(history))
    else:
        content.extend(_image_items([rgb_history[-1]]))

    content.append({"type": "text", "text": "and an image of the current observation"})
    content.extend(_image_items([rgb_history[-1]]))

    suffix = prompt_template.format(instruction).split("current observation", 1)[1]
    content.append({"type": "text", "text": suffix})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def _image_items(images: List[Image.Image]) -> list[dict]:
    return [
        {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{encode_image_base64(image)}",
        }
        for image in images
    ]

