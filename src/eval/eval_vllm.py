import json
import numpy as np
from habitat import Env
from habitat.core.agent import Agent
from tqdm import trange
import os
import io
import base64
import re
from tqdm import tqdm
import cv2
import imageio
from habitat.utils.visualizations import maps
import random
from types import SimpleNamespace
import argparse, habitat
from habitat_extensions import measures, task
from habitat_baselines.config.default import get_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from PIL import Image, ImageFont, ImageDraw
from qwen_vl_utils import process_vision_info
import multiprocessing as mp
import time, math
from openai import OpenAI


def encode_image_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


SYSTEM_PROMPT = "You are a helpful assistant."

BASE_PROMPT_TEMPLATE = "Imagine you are a robot programmed for navigation tasks. "\
    "You have been given a video of historical observations and an image of the current observation. "\
    "Your assigned task is: '{}'. Analyze this series of images to decide your next move, "\
    "which could involve turning left or right by a specific degree or moving forward a certain distance."

STOP_HINT_PROMPT_TEMPLATE = BASE_PROMPT_TEMPLATE + " You may answer stop when the goal has been reached."

SR_STOP_PROMPT_TEMPLATE = "Imagine you are a robot programmed for navigation tasks. "\
    "You have been given a video of historical observations and an image of the current observation. "\
    "Your assigned task is: '{}'. Analyze this series of images to decide your next move. "\
    "Available actions are: stop; move forward by a distance in cm; turn left by degrees; turn right by degrees. "\
    "If the current view already satisfies the destination description or reaches the final landmark, choose stop instead of moving on. "\
    "Otherwise choose the safest next one or two actions that continue following the instruction. "\
    "Respond only with comma-separated actions in these formats: stop, forward <number> cm, turn left <number> degree, turn right <number> degree."


def get_prompt_template(prompt_style):
    if prompt_style == "baseline":
        return BASE_PROMPT_TEMPLATE
    if prompt_style == "stop_hint":
        return STOP_HINT_PROMPT_TEMPLATE
    if prompt_style == "sr_stop":
        return SR_STOP_PROMPT_TEMPLATE
    raise ValueError(f"Unsupported prompt style: {prompt_style}")

def seed_all():
    np.random.seed(41)
    random.seed(41)


def append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def evaluate_agent(result_queue, api_key, base_url, config, dataset, result_path, num_generations,
                    forward_distance, turn_angle, max_action_history, resolution_ratio, prompt_style,
                    temperature, max_episodes) -> None:
 
    env = Env(config.habitat, dataset)

    agent = NaVIDA_Agent(
        api_key, 
        base_url, 
        result_path, 
        forward_distance, 
        turn_angle, 
        max_action_history, 
        resolution_ratio, 
        num_generations,
        prompt_style,
        temperature)

    num_episodes = len(env.episodes)
    if max_episodes is not None:
        num_episodes = min(num_episodes, max_episodes)
    
    EARLY_STOP_ROTATION = 25
    EARLY_STOP_STEPS = 400

    target_key = {"distance_to_goal", "success", "spl", "path_length", "oracle_success","ndtw"}

    count = 0
    
    for _ in range(num_episodes):
        episode_start_time = time.time()

        obs = env.reset()
        iter_step = 0
        agent.reset()

        t_dict = {
            "t_episode": 0,
        }

        continuse_rotation_count = 0
        last_dtg = 999
        if os.path.exists(os.path.join(os.path.join(result_path, "log"),"stats_{}.json".format(env.current_episode.episode_id))):
            if result_queue is not None:
                result_queue.put({"t_episode": 0, "skipped": 1})
            continue
        while not env.episode_over:
            
            info = env.get_metrics()
            
            if info["distance_to_goal"] != last_dtg:
                last_dtg = info["distance_to_goal"]
                continuse_rotation_count=0
            else :
                continuse_rotation_count +=1 
            
            
            action = agent.act(obs, info, env.current_episode.episode_id)

            if continuse_rotation_count > EARLY_STOP_ROTATION or iter_step>EARLY_STOP_STEPS:
                action = {"action": 0}

            
            iter_step+=1
            obs = env.step(action)
            
        info = env.get_metrics()
        result_dict = dict()
        result_dict = {k: info[k] for k in target_key if k in info}
        result_dict["id"] = env.current_episode.episode_id
        count+=1

        with open(os.path.join(os.path.join(result_path, "log"),"stats_{}.json".format(env.current_episode.episode_id)), "w") as f:
            json.dump(result_dict, f, indent=4)
        
        t_dict["t_episode"] = time.time() - episode_start_time
        if result_queue is not None:
            result_queue.put(t_dict)

class NaVIDA_Agent(Agent):
    def __init__(self, api_key, base_url, result_path, forward_distance, 
                    turn_angle, max_action_history, resolution_ratio, num_generations = 1,
                    prompt_style="baseline", temperature=0.2, require_map=True):
        
        print("Initialize NaVIDA")
        
        self.result_path = result_path
        self.require_map = require_map
        self.forward_distance = forward_distance
        self.turn_angle = turn_angle
        self.resolution_ratio = resolution_ratio
        self.max_action_history = max_action_history
        self.num_generations = num_generations
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "log"), exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "video"), exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "trace"), exist_ok=True)

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = self.client.models.list().data[0].id
        
        self.sampling_params = SimpleNamespace(
            n=1,
            temperature=temperature,
            max_tokens=512, # i.e. max_completion_tokens
            top_p=1.0,
        )

        self.promt_template = get_prompt_template(prompt_style)
        self.history_rgb_tensor = None
        
        self.rgb_list = []
        self.topdown_map_list = []
        self.conversations = []
        self.conversations.append({
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]})

        self.reset()

    def uniform_sample_with_ends(self, data, n):
        # n > 2
        if len(data) <= n:
            return data

        indices = [round(i * (len(data) - 1) / (n - 1)) for i in range(n)]
        return [data[i] for i in indices]


    def predict_inference(self):

        outputs = self.client.chat.completions.create(
            messages=self.conversations,
            model=self.model,
            max_completion_tokens=self.sampling_params.max_tokens,
            temperature=self.sampling_params.temperature,
            top_p=self.sampling_params.top_p,
        )
        output_text = outputs.choices[0].message.content
        output_text = output_text.strip()
        
        return output_text

    def _message_text(self, content):
        text_parts = []
        for item in content:
            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif item.get("type") == "image_url":
                text_parts.append("<image>")
        return "\n".join(text_parts)

    def _build_step_record(
        self,
        step_idx,
        observations,
        info,
        raw_output,
        chosen_action,
        action_source,
    ):
        record = {
            "episode_id": self.episode_id,
            "step_idx": step_idx,
            "distance_to_goal": info.get("distance_to_goal"),
            "success": info.get("success"),
            "raw_output": raw_output,
            "chosen_action": chosen_action,
            "action_source": action_source,
            "history_frames": len(self.rgb_list),
            "pending_action_queue": list(self.pending_action_list),
        }
        if step_idx == 0:
            record["instruction"] = observations["instruction"]["text"]
            record["user_prompt"] = (
                self._message_text(self.conversations[-1]["content"])
                if len(self.conversations) > 1
                else None
            )
        return record

    def _log_step(self, record):
        append_jsonl(
            os.path.join(self.result_path, "trace", f"trace_{self.episode_id}.jsonl"),
            record,
        )

    def extract_multi_result(self, output):
        sub_actions = [item for item in re.split(r'\s*,\s*', output.strip()) if item]
        if len(sub_actions) == 0:
            sub_actions = [output]
        result = []
        for sub_action in sub_actions:
            action_index, numeric = self.extract_result(sub_action)
            result.append([action_index, numeric])
        return result

    def extract_result(self, output):
        # id: 0-stop, 1 move forward, 2 turn left, 3 turn right

        output_match = re.search(r'<answer>(.*?)</answer>', output)
        output = output_match.group(1).strip() if output_match else output.strip()

        output = output.lower()
        if "stop" in output:
            return 0, None
        elif "forward" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return 1, self.forward_distance
            match = match.group()
            return 1, float(match)
        elif "left" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return 2, self.turn_angle
            match = match.group()
            return 2, float(match)
        elif "right" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return 3, self.turn_angle
            match = match.group()
            return 3, float(match)
        return None, None
    

    def addtext(self, image, instuction, navigation):
        h, w = image.shape[:2]
        new_height = h + 150
        new_image = np.zeros((new_height, w, 3), np.uint8)
        new_image.fill(255)  
        new_image[:h, :w] = image

        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(instuction, font, 0.5, 2)[0]
        textY = h + (50 + textsize[1]) // 2

        y_line = textY + 0 * textsize[1]

        words = instuction.split(' ')
        max_width = new_image.shape[1]
        x = 10
        line = ""

        for word in words:

            test_line = line + ' ' + word if line else word
            test_line_size, _ = cv2.getTextSize(test_line, font, 0.5, 2)

            if test_line_size[0] > image.shape[1] - x:
                cv2.putText(new_image, line, (x, y_line ), font, 0.5, (0, 0, 0), 2)
                line = word
                y_line += textsize[1]+5
            else:
                line = test_line

        if line:
            cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)
        y_line = y_line + 1 * textsize[1] + 10
        new_image = cv2.putText(new_image, navigation, (x, y_line), font, 0.5, (0, 0, 0), 2)

        return new_image

    def action_id_to_str(self,action_id):
        # id: 0-stop, 1 move forward, 2 turn left, 3 turn right
        if action_id == 0:
            return "stop"
        elif action_id == 1:
            return "forward"
        elif action_id == 2:
            return "turn left"
        elif action_id == 3:
            return "turn right"
        else:
            raise ValueError(f"Invalid action ID: {action_id}")
        
    def reset(self):       
        if self.require_map:
            if len(self.topdown_map_list)!=0:
                output_video_path = os.path.join(self.result_path, "video","{}.gif".format(self.episode_id))

                imageio.mimsave(output_video_path, self.topdown_map_list)

        self.topdown_map_list = []

        self.pending_action_list = []
        self.rgb_list = []

        self.conversations = []
        self.conversations.append({
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]})
        self.step_idx = 0
        
    def act(self, observations, info, episode_id):

        self.episode_id = episode_id
        rgb = observations["rgb"]
        if self.resolution_ratio < 1:
            rgb = cv2.resize(rgb,(0,0),fx=self.resolution_ratio,fy=self.resolution_ratio)
        rgb_ = Image.fromarray(rgb.astype('uint8')).convert('RGB')
        rgb_ = rgb_.resize((308,252))
        self.rgb_list.append(rgb_)
        # do not cut down rgb list while using uniform sampling
        if len(self.rgb_list) > self.max_action_history:
            self.rgb_list = self.rgb_list[1:]

        if self.require_map:
            top_down_map = maps.colorize_draw_agent_and_fit_to_height(info["top_down_map"], rgb.shape[0])
            output_im = np.concatenate((rgb, top_down_map), axis=1)

        if len(self.pending_action_list) != 0 :
            temp_action = self.pending_action_list.pop(0)
            
            if self.require_map:
                img = self.addtext(output_im, observations["instruction"]["text"], "Pending action: {}".format(temp_action))
                self.topdown_map_list.append(img)
            chosen_action = {"action": temp_action}
            self._log_step(
                self._build_step_record(
                    step_idx=self.step_idx,
                    observations=observations,
                    info=info,
                    raw_output=None,
                    chosen_action=chosen_action,
                    action_source="pending_queue",
                )
            )
            self.step_idx += 1
            return chosen_action

        # for observation1+observation2 action style
        self.conversations = self.conversations[:1]
        content = []

        content.append({"type": "text", "text": 'Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations'})
        if len(self.rgb_list) > 1:
            content.extend([{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(item)}"}} for item in self.uniform_sample_with_ends(self.rgb_list[:-1],8)])
        else:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(self.rgb_list[-1])}"}})
        content.append({"type": "text", "text": 'and an image of the current observation'})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(self.rgb_list[-1])}"}})
        item = self.promt_template.format(observations["instruction"]["text"]).split('current observation')
        content.append({"type": "text", "text": item[1]})


        self.conversations.append({
                "role": "user",
                "content": content
            })

        navigation = self.predict_inference()
        raw_navigation = navigation
        
        if self.require_map:
            img = self.addtext(output_im, observations["instruction"]["text"], navigation)
            self.topdown_map_list.append(img)
        
        result = self.extract_multi_result(navigation)
        parsed_actions = [
            {"action_id": action_index, "value": numeric}
            for action_index, numeric in result
        ]

        select_action_idx = 2

        result = result[:select_action_idx]
        for action_index,numeric in result:

            if action_index == 0:
                self.pending_action_list.append(0)
            elif action_index == 1:
                for _ in range(min(3, round(numeric/self.forward_distance))):
                    self.pending_action_list.append(1)

            elif action_index == 2:
                for _ in range(min(3,round(numeric/self.turn_angle))):
                    self.pending_action_list.append(2)

            elif action_index == 3:
                for _ in range(min(3,round(numeric/self.turn_angle))):
                    self.pending_action_list.append(3)
            
            if action_index is None or len(self.pending_action_list)==0:
                print('random select an action')
                action_index = random.randint(1, 3)
                self.pending_action_list.append(action_index)

        chosen_action = {"action": self.pending_action_list.pop(0)}
        self._log_step(
            self._build_step_record(
                step_idx=self.step_idx,
                observations=observations,
                info=info,
                raw_output=raw_navigation,
                chosen_action=chosen_action,
                action_source="model",
            )
        )
        self.step_idx += 1
        return chosen_action


def main():
    seed_all()
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp-config",type=str,required=True,help="path to config yaml containing info about experiment")
    parser.add_argument("--split-num",type=int,required=True,help="chunks of evluation")
    parser.add_argument("--split-id",type=int,default=None,help="optional split ID; when set, run only this split in the current process")
    parser.add_argument("--resolution-ratio",type=float,help="location of model weights",default=0.5)
    parser.add_argument("--result-path",type=str,required=True,help="location to save results")
    parser.add_argument("--forward-distance",type=int,help="distance that one forward action takes",default=25)
    parser.add_argument("--turn-angle",type=int,help="angle that one turn action takes",default=15)
    parser.add_argument("--max-action-history",type=int,help="the maximum num of action history",default=10)
    parser.add_argument("--num-generations",type=int,help="whether use video or multi image",default=1)
    parser.add_argument("--prompt-style", choices=["baseline", "stop_hint", "sr_stop"], default="baseline",
                        help="prompt template used for navigation decisions")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="sampling temperature for vLLM chat completions")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="optional maximum number of episodes to evaluate in each split")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_API_BASE")
    assert api_key is not None and base_url is not None

    config = get_config(args.exp_config)
    with habitat.config.read_write(config):
        # self.config.habitat.task.measurements.success.success_distance=3.0
        config.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=1024,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=True,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=90,
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )
            
    dataset = habitat.datasets.make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)
    dataset_splits = dataset.get_splits(args.split_num, allow_uneven_splits=True)

    if args.split_id is not None:
        evaluate_agent(None, api_key, base_url, config, dataset_splits[args.split_id], args.result_path,
                args.num_generations, args.forward_distance, args.turn_angle,
                args.max_action_history, args.resolution_ratio, args.prompt_style,
                args.temperature, args.max_episodes)
        return

    num_episodes = len(dataset.episodes)
    if args.max_episodes is not None:
        num_episodes = sum(min(len(split.episodes), args.max_episodes) for split in dataset_splits)

    manager = mp.Manager()
    result_queue = manager.Queue()
    processes = []
    for i in range(args.split_num):
        worker_args = (result_queue, api_key, base_url, config, dataset_splits[i], args.result_path,
                args.num_generations, args.forward_distance, args.turn_angle, 
                args.max_action_history, args.resolution_ratio, args.prompt_style,
                args.temperature, args.max_episodes)
        p = mp.Process(target=evaluate_agent, args=worker_args, daemon=True)
        p.start()
        processes.append(p)

    with tqdm(total=num_episodes, desc="Evaluating") as pbar:
        for _ in range(num_episodes):
            result = result_queue.get()
            pbar.update(1)
            pbar.set_postfix(**result)
    
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
