import base64
import io
import json
import urllib.request
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


DEFAULT_SIGLIP_PATH = "/data1/dataset/embAI_sup/siglip-so400m-patch14-384"


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = vector.astype(np.float32)
    norm = np.linalg.norm(vector)
    if norm <= 1e-6:
        return vector
    return vector / norm


class SiglipImageEncoder:
    def __init__(self, model_path: str = DEFAULT_SIGLIP_PATH, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.processor = None
        self.model = None

    def _load(self):
        if self.model is not None:
            return

        import torch
        from transformers import AutoImageProcessor, SiglipVisionModel

        self.processor = AutoImageProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
            use_fast=False,
        )
        self.model = SiglipVisionModel.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.torch = torch

    def encode(self, image) -> np.ndarray:
        self._load()
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with self.torch.inference_mode():
            outputs = self.model(**inputs)
            if getattr(outputs, "pooler_output", None) is not None:
                embedding = outputs.pooler_output[0]
            else:
                embedding = outputs.last_hidden_state[0].mean(dim=0)

        return _normalize(embedding.detach().float().cpu().numpy())


class SiglipMultimodalEncoder:
    def __init__(self, model_path: str = DEFAULT_SIGLIP_PATH, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.image_processor = None
        self.tokenizer = None
        self.model = None

    def _load(self):
        if self.model is not None:
            return

        import torch
        from transformers import AutoImageProcessor, AutoTokenizer, SiglipModel

        self.image_processor = AutoImageProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
            use_fast=False,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            use_fast=False,
        )
        self.model = SiglipModel.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.torch = torch

    def encode(self, image) -> np.ndarray:
        self._load()
        inputs = self.image_processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with self.torch.inference_mode():
            embedding = self.model.get_image_features(pixel_values=pixel_values)[0]
        return _normalize(embedding.detach().float().cpu().numpy())

    def encode_text(self, text: str) -> np.ndarray:
        self._load()
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            embedding = self.model.get_text_features(**inputs)[0]
        return _normalize(embedding.detach().float().cpu().numpy())


class RemoteSiglipImageEncoder:
    def __init__(self, server_url: str, timeout: float = 30.0):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            self.server_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.opener.open(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def encode(self, image) -> np.ndarray:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        result = self._post("/encode", {
            "image": base64.b64encode(buffer.getvalue()).decode("utf-8"),
        })
        return _normalize(np.asarray(result["embedding"], dtype=np.float32))

    def encode_text(self, text: str) -> np.ndarray:
        result = self._post("/encode_text", {"text": text})
        return _normalize(np.asarray(result["embedding"], dtype=np.float32))


@dataclass
class TopoNode:
    node_id: int
    first_step: int
    last_step: int
    visits: int
    embedding: np.ndarray
    caption: Optional[str] = None


@dataclass
class TopoEdge:
    src: int
    dst: int
    count: int
    action_summary: Optional[str]
    first_step: int
    last_step: int


class TopoMemoryGraph:
    def __init__(
        self,
        encoder,
        sim_threshold: float = 0.84,
        max_nodes: int = 80,
        text_max_lines: int = 4,
        semantic: bool = False,
        node_captioner: Optional[Callable] = None,
        include_avoid_hint: bool = True,
    ):
        self.encoder = encoder
        self.sim_threshold = sim_threshold
        self.max_nodes = max_nodes
        self.text_max_lines = text_max_lines
        self.semantic = semantic
        self.node_captioner = node_captioner
        self.include_avoid_hint = include_avoid_hint
        self.reset()

    def reset(self):
        self.nodes: List[TopoNode] = []
        self.edges: Dict[Tuple[int, int], TopoEdge] = {}
        self.current_node_id: Optional[int] = None
        self.next_node_id = 0
        self.recent_path: List[int] = []
        self.last_observation = None

    def observe(self, image, step_idx: int, transition_action: Optional[str] = None):
        embedding = self.encoder.encode(image)
        match_node, match_similarity = self._best_match(embedding)

        is_revisit = match_node is not None and match_similarity >= self.sim_threshold
        previous_node_id = self.current_node_id

        if is_revisit:
            node = match_node
            node.visits += 1
            node.last_step = step_idx
            node.embedding = _normalize(0.9 * node.embedding + 0.1 * embedding)
            if self.semantic and not node.caption:
                node.caption = self._caption_image(image)
        else:
            node = TopoNode(
                node_id=self.next_node_id,
                first_step=step_idx,
                last_step=step_idx,
                visits=1,
                embedding=embedding,
                caption=self._caption_image(image) if self.semantic else None,
            )
            self.nodes.append(node)
            self.next_node_id += 1

        self.current_node_id = node.node_id
        if previous_node_id is not None and previous_node_id != node.node_id:
            self._update_edge(previous_node_id, node.node_id, step_idx, transition_action)

        if not self.recent_path or self.recent_path[-1] != node.node_id:
            self.recent_path.append(node.node_id)
            self.recent_path = self.recent_path[-6:]

        self._prune()

        self.last_observation = {
            "node_count": len(self.nodes),
            "current_node": node.node_id,
            "matched_node": match_node.node_id if match_node is not None else None,
            "match_similarity": float(match_similarity) if match_similarity is not None else None,
            "is_revisit": bool(is_revisit),
            "visits": node.visits,
            "first_step": node.first_step,
            "last_step": node.last_step,
            "steps_since_first_visit": step_idx - node.first_step,
            "current_caption": node.caption,
            "matched_caption": match_node.caption if match_node is not None else None,
        }
        return self.last_observation

    def build_prompt_text(self) -> str:
        if self.last_observation is None:
            return ""
        if self.semantic:
            return self._build_semantic_prompt_text()
        return self._build_basic_prompt_text()

    def _build_basic_prompt_text(self) -> str:
        obs = self.last_observation
        current_node = obs["current_node"]
        lines = [
            f"Topological memory: {obs['node_count']} places recorded; current place is {current_node}."
        ]

        if obs["is_revisit"] and obs["steps_since_first_visit"] > 0:
            lines.append(
                "Current view revisits place "
                f"{current_node}, first seen {obs['steps_since_first_visit']} decisions ago, "
                f"visits={obs['visits']}, similarity={obs['match_similarity']:.2f}."
            )
        elif obs["node_count"] > 1:
            lines.append(f"Current view is a new place with nearest similarity {obs['match_similarity']:.2f}.")

        if len(self.recent_path) >= 2:
            route = " -> ".join(f"place {node_id}" for node_id in self.recent_path[-5:])
            lines.append(f"Recent route: {route}.")

        if self.include_avoid_hint and obs["is_revisit"] and obs["steps_since_first_visit"] > 0:
            lines.append("Avoid repeating the same route unless the instruction requires returning.")

        return "\n".join(lines[: self.text_max_lines])

    def _build_semantic_prompt_text(self) -> str:
        obs = self.last_observation
        current_label = self._node_label(obs["current_node"])
        lines = [f"Semantic topological memory: {obs['node_count']} places recorded."]

        if obs["is_revisit"] and obs["steps_since_first_visit"] > 0:
            lines.append(
                f"Current view matches {current_label}, first seen {obs['steps_since_first_visit']} decisions ago, "
                f"visits={obs['visits']}, similarity={obs['match_similarity']:.2f}."
            )
        elif obs["node_count"] > 1:
            lines.append(
                f"Current view is a new place: {current_label}; "
                f"nearest known place similarity={obs['match_similarity']:.2f}."
            )
        else:
            lines.append(f"Current view is {current_label}.")

        if len(self.recent_path) >= 2:
            route = " -> ".join(self._node_label(node_id) for node_id in self.recent_path[-4:])
            lines.append(f"Recent route: {route}.")

        if obs["is_revisit"] and obs["steps_since_first_visit"] > 0:
            lines.append("The recent route has returned to a previously seen visual place.")

        return "\n".join(lines[: self.text_max_lines])

    def _node_label(self, node_id: int) -> str:
        node = self._get_node(node_id)
        if node is None or not node.caption:
            return f"place {node_id}"
        return f"place {node_id} ({node.caption})"

    def _get_node(self, node_id: int):
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def _caption_image(self, image) -> Optional[str]:
        if self.node_captioner is None:
            return None
        caption = self.node_captioner(image)
        if caption is None:
            return None
        caption = " ".join(str(caption).strip().split())
        if caption.endswith("."):
            caption = caption[:-1]
        return caption[:160]

    def _best_match(self, embedding: np.ndarray):
        if not self.nodes:
            return None, None
        similarities = [float(np.dot(node.embedding, embedding)) for node in self.nodes]
        best_index = int(np.argmax(similarities))
        return self.nodes[best_index], similarities[best_index]

    def _update_edge(self, src: int, dst: int, step_idx: int, action_summary: Optional[str]):
        key = (src, dst)
        if key not in self.edges:
            self.edges[key] = TopoEdge(
                src=src,
                dst=dst,
                count=1,
                action_summary=action_summary,
                first_step=step_idx,
                last_step=step_idx,
            )
            return

        edge = self.edges[key]
        edge.count += 1
        edge.last_step = step_idx
        if action_summary:
            edge.action_summary = action_summary

    def _prune(self):
        if len(self.nodes) <= self.max_nodes:
            return

        ranked_nodes = sorted(
            self.nodes,
            key=lambda node: (
                node.node_id == self.current_node_id,
                node.last_step,
                node.visits,
            ),
            reverse=True,
        )
        keep_ids = {node.node_id for node in ranked_nodes[: self.max_nodes]}
        self.nodes = [node for node in self.nodes if node.node_id in keep_ids]
        self.edges = {
            key: edge
            for key, edge in self.edges.items()
            if edge.src in keep_ids and edge.dst in keep_ids
        }
        self.recent_path = [node_id for node_id in self.recent_path if node_id in keep_ids]
