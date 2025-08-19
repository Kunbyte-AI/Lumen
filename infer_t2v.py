import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import json
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm
import random
random.seed(42)
from PIL import Image
import decord

def worker(rank, gpu_id, video_paths_chunk, configs):
    import torch
    from diffsynth import ModelManager, WanVideoPipeline, save_video

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_manager = ModelManager(device="cpu") # 1.3B: device=cpu(先加载到cpu上): 占6G显存, device=device: 占16G显存
    # 14B: 会借用0卡加载参数, 之后再到各卡上推理, 约占36G, 一个视频约10min
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
    if 'wan14b' in configs['save_dir_path']:
        model_manager.load_models(
            [
                configs['wan_dit_path'] if configs['wan_dit_path'] else 'ckpt/Wan2.1-Fun-14B-Control/diffusion_pytorch_model.safetensors',
                'ckpt/Wan2.1-Fun-1.3B-Control/Wan2.1_VAE.pth',
                'ckpt/Wan2.1-Fun-1.3B-Control/models_t5_umt5-xxl-enc-bf16.pth',
                'ckpt/Wan2.1-Fun-1.3B-Control/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth',
            ],
            torch_dtype=torch.bfloat16, # float8_e4m3fn fp8量化; bfloat16
        )
    else:
        model_manager.load_models(
            [
                configs['wan_dit_path'] if configs['wan_dit_path'] else 'ckpt/Wan2.1-Fun-1.3B-Control/diffusion_pytorch_model.safetensors',
                'ckpt/Wan2.1-Fun-1.3B-Control/Wan2.1_VAE.pth',
                'ckpt/Wan2.1-Fun-1.3B-Control/models_t5_umt5-xxl-enc-bf16.pth',
                'ckpt/Wan2.1-Fun-1.3B-Control/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth',
            ],
            torch_dtype=torch.bfloat16,
        )
    
    if configs['lora_path']:
        model_manager.load_lora(configs['lora_path'], lora_alpha=configs['lora_alpha'])
    
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)

    num_frames, height, width = configs['num_frames'], configs['height'], configs['width']

    def get_video(video_path, mask_path, bg_value=0):
        video_reader = decord.VideoReader(video_path, width=width, height=height)
        mask_reader = decord.VideoReader(mask_path, width=width, height=height)
        
        assert len(video_reader) == len(mask_reader), f"视频帧数和mask帧数不匹配: {len(video_reader)} != {len(mask_reader)}"
        
        # total_frames<num_frames时, 会自动补全
        select_frames = np.linspace(0, len(video_reader)-1, num_frames, dtype=int)
        
        video_frames = video_reader.get_batch(select_frames).asnumpy().astype(np.uint8)
        mask_frames = mask_reader.get_batch(select_frames).asnumpy().astype(np.uint8)
        masked_v = np.where(mask_frames >= 127, video_frames, bg_value)
        mask_f1 = mask_frames[0].copy()

        return video_frames, masked_v, mask_f1

    def concatenate_images_horizontally(images_list, output_type="pil"):
        concatenated_images = []
        length = len(images_list[0])  # 帧数
        
        # 找出每个视频列表中图像的最大高度
        max_height = 0
        for item in images_list:
            # for img in item:
            img = item[0]
            img_array = np.array(img)
            max_height = max(max_height, img_array.shape[0])
        
        for i in range(length):
            # 获取当前帧中每个视频的图像并调整为相同高度
            current_frames = []
            for item in images_list:
                img_array = np.array(item[i])
                h, w, c = img_array.shape
                
                # 如果高度小于最大高度，创建一个0填充的背景并将图像放在其中
                if h < max_height:
                    # 创建黑色背景
                    padded_img = np.zeros((max_height, w, c), dtype=np.uint8)
                    # 将原图放在顶部居中
                    padded_img[:h, :, :] = img_array
                    current_frames.append(padded_img)
                else:
                    current_frames.append(img_array)
            
            # 水平拼接当前帧中的所有图像
            concatenated_img = np.concatenate(current_frames, axis=1)
            
            # 转换输出格式
            if output_type == "pil":
                concatenated_img = Image.fromarray(concatenated_img)
            elif output_type == "np":
                pass
            else:
                raise NotImplementedError
                
            concatenated_images.append(concatenated_img)
        
        return concatenated_images

    for i, video_path in enumerate(video_paths_chunk):
        video_dir, video_name = os.path.split(video_path)

        mask_path = video_path.replace(configs['video_dir_path'], configs['mask_dir_path'])
        ori_v, fg_v, _ = get_video(video_path, mask_path)

        print(f"[GPU {gpu_id}] Processing {i+1}/{len(video_paths_chunk)}: {video_dir}/{video_name}")
        
        save_dir_path = os.path.join(configs['save_dir_path'], video_dir.split('/')[-1])
        os.makedirs(save_dir_path, exist_ok=True)
        
        save_path = os.path.join(save_dir_path, video_name)
        video_i = configs['select_indices'][rank] + i
        # save_path = os.path.join(save_dir_path, f'{video_i+1:03d}.mp4')
        
        if os.path.exists(save_path): continue
        
        prompt = configs['prompts'][ video_i % len(configs['prompts']) ]

        video = pipe(
            prompt=prompt,
            # negative_prompt = 'Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards',
            negative_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走',
            num_inference_steps = 50, 
            control_video=fg_v,
            height=height, width=width, num_frames=num_frames,
            seed=-1, tiled=True,
        )

        concat_video = concatenate_images_horizontally([ori_v, video])
        save_video(concat_video, save_path, fps=16, quality=7)
        # save_video(video, save_path, fps=16, quality=7)


def main(): # python infer_t2v.py
    gpu_ids = [ 0,1,2,3, 4,5,6,7 ]
    # gpu_ids = [ 3 ]

    # wan_dit_path = None
    wan_dit_path = 'ckpt/Lumen/Lumen-T2V-1.3B-V1.0.ckpt'
    save_dir_path = 'test_res/wan1.3b/'
    lora_path = None
    lora_alpha = 0

    # wan_dit_path = None
    # lora_alpha = 1
    # lora_path = 'train_res/wan14b/...'
    # save_dir_path = 'test_res/wan14b/...'

    
    # 若wan_dit_path为list, 加载其中所有.safetensors文件路径为list
    if wan_dit_path and os.path.isdir(wan_dit_path):
        wan_dit_path = [os.path.join(wan_dit_path, f) for f in os.listdir(wan_dit_path) if f.endswith('.safetensors')]
        wan_dit_path.sort()

    # test pachong
    video_dir_path = 'test/pachong_test/video/single_70'
    mask_dir_path = 'test/pachong_test/video_rmbg_msk/single'
    # video_paths = [os.path.join(video_dir_path, f) for f in os.listdir(video_dir_path) ]
    video_names = [ 191947, 922930, 1217498, 1302135, 1371894, 
                    1628805, 1873403, 2080723, 2259812, 2445920, 
                    2639840, 2779867, 2974076 ] # 13
    video_names = video_names * 2
    video_paths = [ os.path.join(video_dir_path, f'{name}.mp4') for name in video_names ]
    
    print('len(video_paths):', len(video_paths))

    num_frames, height, width = 49, 480, 832

    bg_prompt_path = 'my_data/zh_en_short_prompts.txt'

    with open(bg_prompt_path, 'r') as f:
        bg_prompts = f.readlines()
    prompts = [bg.strip() for bg in bg_prompts if bg.strip()]  # 去除空行
    prompts = prompts[ : 26 ] # len(prompts) // 2 = 26

    print(f"Loaded *{len(prompts)}* background prompts from {bg_prompt_path}")

    num_gpus = len(gpu_ids)
    # 在video_paths里尽可能均匀选取num_gpus-1个中间位置, 作为分割点
    select_indices = np.linspace(0, len(video_paths), num_gpus+1, dtype=int)
    v2_chunks = [video_paths[select_indices[i]:select_indices[i+1]] for i in range(num_gpus)]

    configs = {
        'wan_dit_path': wan_dit_path,
        'lora_path': lora_path,
        'lora_alpha': lora_alpha,
        'num_frames': num_frames,
        'height': height,
        'width': width,
        'select_indices': select_indices,
        'video_dir_path': video_dir_path,
        'mask_dir_path': mask_dir_path,
        'save_dir_path': save_dir_path,
        'prompts': prompts,
    }

    mp.set_start_method('spawn', force=True)
    processes = []
    for rank in range(num_gpus):
        p = mp.Process(target=worker, args=(rank, gpu_ids[rank], v2_chunks[rank], configs))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()

