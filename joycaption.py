# https://github.com/fpgaminer/joycaption

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import decord
import os
from tqdm import tqdm
import torch.multiprocessing as mp
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"
gpu_ids = [0,1,2,3]
num_process_per_gpu = 1

desc_prompt = 'Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt in the background, lighting and foregorund people. 20 words or less.'
model_path = "ckpt/llama-joycaption-beta-one-hf-llava"

video_paths = []
video_dir = 'my_data/pachong/video_len49'
for root, dirs, files in os.walk(video_dir):
    for file in files:
        if file.endswith('.mp4'):
            video_paths.append(os.path.join(root, file))
video_paths.sort()
res_save_dir = 'my_data/caption/r2_cap_res_en_short' # python joycaption.py

# video_dir = 'my_data'
# with open('my_data/.../1real_video_paths.json', 'r') as f:
#     video_paths = json.load(f)
# video_paths = [ os.path.join(video_dir, video_path) for video_path in video_paths ]
# res_save_dir = 'my_data/caption/joy_1495'

print('len video_paths:', len(video_paths))
os.makedirs(res_save_dir, exist_ok=True)

@torch.no_grad()
def run_inference(rank, gpu_id, video_paths_subset):
    # Load JoyCaption
    # bfloat16 is the native dtype of the LLM used in JoyCaption (Llama 3.1); device_map=0 loads the model into the first GPU
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(device)
    processor = AutoProcessor.from_pretrained(model_path)
    llava_model = LlavaForConditionalGeneration.from_pretrained(model_path, torch_dtype="bfloat16", device_map={"": device}, )
    llava_model.eval()

    res_pairs = {}
    res_save_path = f'{res_save_dir}/rank{rank}.json'
    
    for video_path in tqdm(video_paths_subset, desc="Processing videos", ncols=100):
        # with torch.no_grad():
        # Load image
        # image = Image.open(IMAGE_PATH)
        video_f1 = decord.VideoReader(video_path).get_batch(range(1)).asnumpy().astype('uint8')
        image = Image.fromarray(video_f1[0])

        # Build the conversation
        convo = [
            {
                "role": "system",
                "content": "You are a helpful image captioner.",
            },
            {
                "role": "user",
                "content": desc_prompt,
            },
        ]

        # Format the conversation
        # WARNING: HF's handling of chat's on Llava models is very fragile.  This specific combination of processor.apply_chat_template(), and processor() works
        # but if using other combinations always inspect the final input_ids to ensure they are correct.  Often times you will end up with multiple <bos> tokens; if not careful, which can make the model perform poorly.
        # HF在Llava模型上对CHAT的处理非常脆弱。 processor.apply_chat_template（）和processor（）的特定组合可以work
        # 但是，如果使用其他组合，请务必检查最终输入_ID以确保它们正确。通常，您最终会得到多个<bos>令牌
        # 如果不小心，这会使模型的性能不佳。
        convo_string = processor.apply_chat_template(convo, tokenize = False, add_generation_prompt = True)
        assert isinstance(convo_string, str)

        # Process the inputs
        inputs = processor(text=[convo_string], images=[image], return_tensors="pt").to(device)
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

        # Generate the captions
        generate_ids = llava_model.generate(
            **inputs,
            max_new_tokens=100, # 512
            do_sample=True,
            suppress_tokens=None,
            use_cache=True,
            temperature=0.6,
            top_k=None,
            top_p=0.9,
        )[0]

        # Trim off the prompt
        generate_ids = generate_ids[inputs['input_ids'].shape[1]:]

        # Decode the caption
        caption = processor.tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        caption = caption.strip()
        # print(caption)
        res_pairs[video_path] = caption

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