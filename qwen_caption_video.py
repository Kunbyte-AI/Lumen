# https://github.com/QwenLM/Qwen2.5-VL
# pip install qwen-vl-utils[decord]
# pip install flash-attn opencv-python

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
import torch.multiprocessing as mp
import json
from tqdm import tqdm
import decord
import numpy as np
import base64
import cv2

os.environ["TOKENIZERS_PARALLELISM"] = "false"
model_path = 'ckpt/Qwen2.5-VL-7B-Instruct/'
gpu_ids = [ 4,5,6,7 ] # 0,1,2,3 4,5,6,7
num_process_per_gpu = 1
# python qwen_caption_video.py
# 修改: res_save_dir, prompt, gpu_ids

# video_paths = []
# with open('test/xx.json', 'r') as f:
#     video_pairs = json.load(f)
# # video_paths = list(set(video_pairs.values()))
# video_paths = [ list(video_pair.values())[0] for video_pair in video_pairs ]
# video_paths = list(set(video_paths))
# video_paths.sort()

video_paths = []
video_dir = 'my_data/pachong/video_len49'
for root, dirs, files in os.walk(video_dir):
    for file in files:
        if file.endswith('.mp4'):
            video_paths.append(os.path.join(root, file))
video_paths.sort()


print('len video_paths:', len(video_paths))
res_save_dir = 'my_data/caption/r2_cap_res_zh_short' # zh_long zh_short en_long
os.makedirs(res_save_dir, exist_ok=True)

## desc_prompt = 'Describe the video in short: Foreground people, Background, and especially the effect of background light on foreground people, etc. Stable diffusion prompt style, phrases. Around 20 words.'

# en long
# desc_prompt = 'Describe the video in detail: Foreground people: appearance, movement, clothing; Background: objects, time, location, weather, overall tone; Light: location, type, color, brightness, effect; Camera: angle, distance, movement; Important: the effect of the environment light on the people in the foreground: light, shadow, color, etc. Around 50 words.'

# zh long
# desc_prompt = '从以下几个方面详细描述视频内容: 前景人物(外貌, 动作, 服装), 背景(物体, 时间, 地点, 天气, 整体色调), 光照(光源位置, 类型, 颜色, 强度, 效果), 摄像机(角度, 距离, 运动). 重要: 环境光对前景人物的影响: 光线, 阴影, 颜色等. 60个词左右.'

# zh short
desc_prompt = '从前景人物外貌, 背景内容, 以及背景光对前景的影响这三个方面简要描述视频, 30个词左右.'

# prompt for IC-Light/Light-A-Video/RelightVid style caption
# desc_prompt = '''Describe the video in short, stable diffusion prompt style. Output:
# video caption: a short sentence describing the foreground and background. examples: a car driving on the street; a man in the classroom;
# light caption: examples: sunshine from window; neon light, city sunset over sea; golden time; sci-fi RGB glowing, cyberpunk; natural lighting; warm atmosphere; magic lit; Wong Kar-wai.
# light source: choose in LEFT, RIGHT, TOP, BOTTOM, NONE.
# '''

def run_inference(rank, gpu_id, video_paths_subset):
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(device)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", # pip install flash-attn
        device_map={"": device},
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)

    # res_pairs = []
    res_pairs = {}
    res_save_path = f'{res_save_dir}/rank{rank}.json'

    # for video_path in video_dir_paths_subset:
    for video_path in tqdm(video_paths_subset):
        save_key = video_path
        # save_key = video_path.split('/')[-1]
        # print(f"rank {rank} processing {video_path}")
        video_reader = decord.VideoReader(video_path)
        # 均匀选择10帧
        select_10_frames = np.linspace(0, len(video_reader) - 1, 10).astype(int)
        video_frames = video_reader.get_batch(select_10_frames).asnumpy()
        # Base64 encoded image
        base64_images = []
        for frame in video_frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) #* 转换颜色
            _, buffer = cv2.imencode('.jpg', frame_bgr)
            base64_image = base64.b64encode(buffer).decode('utf-8')
            base64_images.append(base64_image)

        messages = [
            {
                "role": "user",
                "content": [
                    # {"type": "image", "image": image_paths[0]},
                    # {"type": "image", "image": "data:image;base64,/9j/..."},
                    {"type": "image", "image": "data:image;base64," + base64_images[0]},
                    {"type": "image", "image": "data:image;base64," + base64_images[1]},
                    {"type": "image", "image": "data:image;base64," + base64_images[2]},
                    {"type": "image", "image": "data:image;base64," + base64_images[3]},
                    {"type": "image", "image": "data:image;base64," + base64_images[4]},
                    {"type": "image", "image": "data:image;base64," + base64_images[5]},
                    {"type": "image", "image": "data:image;base64," + base64_images[6]},
                    {"type": "image", "image": "data:image;base64," + base64_images[7]},
                    {"type": "image", "image": "data:image;base64," + base64_images[8]},
                    {"type": "image", "image": "data:image;base64," + base64_images[9]},

                    {"type": "text", "text": desc_prompt},
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # Inference
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # res_pairs.append({video_path: output_text[0]})
        res_pairs[save_key] = output_text[0]
    
    with open(res_save_path, 'w') as f:
        json.dump(res_pairs, f, indent=2, ensure_ascii=False)
    print(f"rank {rank} finished, saved to {res_save_path}")

if __name__ == "__main__":
    num_gpus = len(gpu_ids)
    total_processes = num_gpus * num_process_per_gpu
    # 均分数据到每个进程
    chunk_size = (len(video_paths) + total_processes - 1) // total_processes
    chunks = [video_paths[i*chunk_size:(i+1)*chunk_size] for i in range(total_processes)]

    mp.set_start_method('spawn', force=True)
    processes = []
    for proc_idx in range(total_processes):
        gpu_id = gpu_ids[proc_idx // num_process_per_gpu]
        p = mp.Process(target=run_inference, args=(proc_idx, gpu_id, chunks[proc_idx]))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
